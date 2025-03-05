import sys
sys.path.insert(0, '../../')
import numpy as np
import torch
import torch.nn.functional as F
from DGP import generate_all_scenarios
from geomloss import SamplesLoss  # Import geomloss
from model.model import CFRNet

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_and_evaluate_tarnet(model, data, num_epochs=60, batch_size=512,da_weight=20.0, loss_th=1000.0):
    """Training loop for TARNet"""
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    
    X = torch.FloatTensor(data['X'])
    T = torch.LongTensor(data['T'])
    Y = torch.FloatTensor(data['Y'])
 
    #da_loss = SamplesLoss(loss="energy", backend="tensorized")
    da_loss = SamplesLoss(loss="sinkhorn", p=1, blur=0.05, scaling=0.5)

    if len(Y.shape) == 1:
        Y = Y.unsqueeze(-1)
    
    # Split data
    n_train = int(0.8 * len(X))
    train_idx = np.random.choice(len(X), n_train, replace=False)
    test_idx = np.array(list(set(range(len(X))) - set(train_idx)))
    
    X_train, T_train, Y_train = X[train_idx], T[train_idx], Y[train_idx]
    X_test, T_test, Y_test = X[test_idx], T[test_idx], Y[test_idx]
    
    # Create DataLoader
    train_dataset = torch.utils.data.TensorDataset(X_train, T_train, Y_train)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True
    )
    
    optimizer = torch.optim.Adam(model.parameters())
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_pred_loss = 0.0
        epoch_da_loss = 0.0
        num_batches = 0
        
        for batch_x, batch_t, batch_y in train_loader:
            optimizer.zero_grad()
            
            # Forward pass
            data_dict = {"feature": batch_x}
            pred, hypo = model(data_dict)
            
            # Get predictions for actual treatments
            pred_t = torch.gather(pred, 2, batch_t.unsqueeze(1).unsqueeze(2).expand(-1, pred.size(1), 1))
            pred_t = pred_t.squeeze(2)  # [batch, num_outcomes]
            
            # Prediction loss
            pred_loss = F.mse_loss(pred_t, batch_y)
            
            batch_size = pred.shape[0]
            da_losses = []

            treatment_groups = {}
            for i in range(len(batch_t)):
                t = batch_t[i].item()
                if t not in treatment_groups:
                    treatment_groups[t] = []
                treatment_groups[t].append(hypo[i])
                
            # Skip if we don't have enough samples for comparison
            if len(treatment_groups) < 2:
                continue
            
            all_data = torch.cat([torch.stack(treatment_groups[t]) for t in treatment_groups], dim=0)

            # Compare each treatment group against the combined data
            cfr_loss = 0
            num_comparisons = 0
            for t in treatment_groups:
                t_data = torch.stack(treatment_groups[t])
                # Use geomloss instead of custom MMD
                cfr_loss += da_loss(t_data, all_data) / batch_size
                num_comparisons += 1
                
                if num_comparisons > 0:
                    da_losses.append(cfr_loss / num_comparisons)
            
            # Combine losses
            if da_losses:
                da_loss_val = torch.stack(da_losses).mean()
                # Add gradient clipping to prevent explosion
                da_loss_val = torch.clamp(da_loss_val, max=loss_th)
                loss = pred_loss + da_weight * da_loss_val
            else:
                da_loss_val = torch.tensor(0.0)
                loss = pred_loss

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_pred_loss += pred_loss.item()
            epoch_da_loss += da_loss_val.item()
            num_batches += 1

            num_batches += 1
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: total_loss = {epoch_loss/num_batches:.4f}, "
                  f"pred_loss = {epoch_pred_loss/num_batches:.4f}, "
                  f"da_loss = {epoch_da_loss/num_batches:.4f}")
    
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        # Calculate MSE for each treatment-outcome combination
        num_outcomes = Y_test.shape[1]
        num_treatments = len(torch.unique(T))
        mse_matrix = torch.zeros(num_treatments, num_outcomes)
        
        data_dict = {"feature": X_test}
        all_preds, _ = model(data_dict)  # [batch, num_outcomes, num_treatments]
        
        for t in range(num_treatments):
            mask = (T_test == t)
            if mask.any():
                t_pred = all_preds[mask, :, t]  # [batch_t, num_outcomes]
                t_true = Y_test[mask]
                t_mse = F.mse_loss(t_pred, t_true, reduction='none').mean(0)
                mse_matrix[t] = t_mse
        
        # Calculate CATE
        control_pred = all_preds[:, :, 0]  # [batch, num_outcomes]
        cate_dict = {}
        
        # Compare each treatment to control for each outcome
        for m in range(num_outcomes):
            outcome_rmses = []
            for t in range(1, num_treatments):
                pred_cate = all_preds[:, m, t] - control_pred[:, m]
                true_cate = torch.FloatTensor(data['true_cate'][f'outcome{m}'])[test_idx][:, t-1]
                rmse = torch.sqrt(((true_cate - pred_cate) ** 2).mean())
                cate_dict[f'outcome{m}_treatment{t}'] = rmse.item()
                outcome_rmses.append(rmse.item())
            cate_dict[f'outcome{m}_average'] = np.mean(outcome_rmses)
        
        # Calculate treatment-specific averages
        for t in range(1, num_treatments):
            cate_dict[f'treatment{t}_average'] = np.mean([
                cate_dict[f'outcome{m}_treatment{t}'] 
                for m in range(num_outcomes)
            ])
        
        # Calculate overall average
        cate_dict['average'] = np.mean([
            cate_dict[f'outcome{m}_average']
            for m in range(num_outcomes)
        ])
        
        # Print results
        print("\nDetailed MSE Analysis:")
        print("Treatment | " + " | ".join([f"Outcome {i}" for i in range(num_outcomes)]))
        print("-" * (10 + 12 * num_outcomes))
        for t in range(num_treatments):
            mse_values = [f"{mse:.4f}" for mse in mse_matrix[t]]
            print(f"    {t}    | " + " | ".join(mse_values))
        
        print("\nCATE RMSE Analysis:")
        header = "         |" + " | ".join([f" Treatment {t}" for t in range(1, num_treatments)]) + " | Average"
        print(header)
        print("-" * (len(header) + 5))
        
        for m in range(num_outcomes):
            rmse_values = [f"{cate_dict[f'outcome{m}_treatment{t}']:.4f}" for t in range(1, num_treatments)]
            print(f"Outcome {m} | " + " | ".join(rmse_values) + f" | {cate_dict[f'outcome{m}_average']:.4f}")
        
        print("-" * (len(header) + 5))
        avg_values = [f"{cate_dict[f'treatment{t}_average']:.4f}" for t in range(1, num_treatments)]
        print(f"Average  | " + " | ".join(avg_values) + f" | {cate_dict['average']:.4f}")
        
        detailed_results = {
            'overall_mse': mse_matrix.mean().item(),
            'mse_matrix': mse_matrix.tolist(),
            'cate_rmse': cate_dict
        }
    
    return detailed_results

def evaluate_tarnet_all_scenarios():
    """Evaluate TARNet on all synthetic scenarios"""
    scenarios = generate_all_scenarios()
    results = {}
    
    scenario_desc = {
        'A': 'Multiple T, Multiple O, No Conf',
        'B': 'Multiple T, Multiple O, With Conf'
    }
    
    # Define scenario-specific hyperparameters
    scenario_params = {
        'A': {'num_epochs': 120, 'da_weight': 1.0, 'loss_th': 1000.0},
        'B': {'num_epochs': 120, 'da_weight': 20.0, 'loss_th': 1000.0},
    }
    
    print("\nEvaluating CFRNet across all scenarios...")
    print("-" * 60)
    
    for scenario_key, data in scenarios.items():
        try:
            params = scenario_params[scenario_key]
            print(f"\nScenario {scenario_key}: {scenario_desc[scenario_key]}")
            print(f"Epochs: {params['num_epochs']}")
            
            X = data['X']
            T = data['T']
            Y = data['Y']
            
            input_dim = X.shape[1]
            num_treatments = len(np.unique(T))
            num_outcomes = Y.shape[1] if len(Y.shape) > 1 else 1
            
            print(f"Data dimensions: Features={input_dim}, Treatments={num_treatments}, Outcomes={num_outcomes}")
            
            model = CFRNet(
                input_dim=input_dim,
                num_outcome=num_outcomes,
                num_treatment=num_treatments,
                task="regression"
            )
            
            detailed_results = train_and_evaluate_tarnet(
                model, 
                data,
                num_epochs=params['num_epochs'],
                batch_size=512
            )
            results[scenario_key] = detailed_results
            
            print(f"Overall MSE: {detailed_results['overall_mse']:.4f}")
            print(f"Average CATE RMSE: {detailed_results['cate_rmse']['average']:.4f}")
            
        except Exception as e:
            print(f"Error in scenario {scenario_key}: {str(e)}")
            results[scenario_key] = None
    
    return results

if __name__ == "__main__":
    results = evaluate_tarnet_all_scenarios()
    
    # Print summary table
    print("\nSummary of Results")
    print("-" * 80)
    print("Scenario    Description                          Pred MSE    CATE RMSE")
    print("-" * 80)
    for scenario, res in results.items():
        if res is not None:
            desc = {
                'A': 'Multiple T, Multiple O, No Conf',
                'B': 'Multiple T, Multiple O, With Conf'
            }
            print(f"{scenario:<10} {desc[scenario]:<35} {res['overall_mse']:.4f}      {res['cate_rmse']['average']:.4f}") 