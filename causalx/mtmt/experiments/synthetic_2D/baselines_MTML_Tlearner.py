import sys
sys.path.insert(0, '../../')
import numpy as np
import torch
import torch.nn.functional as F
from DGP import generate_all_scenarios
from model.model import Tlearner

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_and_evaluate_tlearner(model, data, num_epochs=60, batch_size=256):
    """Training loop for T-learner"""
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    
    X = torch.FloatTensor(data['X'])
    T = torch.LongTensor(data['T'])
    Y = torch.FloatTensor(data['Y'])
    
    if len(Y.shape) == 1:
        Y = Y.unsqueeze(-1)
    
    # Split data into train and test sets
    n_samples = len(X)
    n_train = int(0.8 * n_samples)
    
    indices = np.random.permutation(n_samples)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]
    
    X_train, T_train, Y_train = X[train_idx], T[train_idx], Y[train_idx]
    X_test, T_test, Y_test = X[test_idx], T[test_idx], Y[test_idx]
   
    # Create DataLoaders
    train_dataset = torch.utils.data.TensorDataset(X_train, T_train, Y_train)
    test_dataset = torch.utils.data.TensorDataset(X_test, T_test, Y_test)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    optimizer = torch.optim.Adam(model.parameters())
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for batch_x, batch_t, batch_y in train_loader:
            optimizer.zero_grad()
            
            # Forward pass
            data_dict = {"feature": batch_x}
            pred = model(data_dict)
            
            # Get predictions for actual treatments
            pred_t = torch.gather(pred, 2, batch_t.unsqueeze(1).unsqueeze(2).expand(-1, pred.size(1), 1))
            pred_t = pred_t.squeeze(2)
            
            # Prediction loss
            loss = F.mse_loss(pred_t, batch_y)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
        
        # Test phase
        model.eval()
        test_loss = 0.0
        test_samples = 0
        
        with torch.no_grad():
            for batch_x, batch_t, batch_y in test_loader:
                data_dict = {"feature": batch_x}
                pred = model(data_dict)
                
                # Calculate MSE only for observed treatments
                pred_t = torch.gather(pred, 2, batch_t.unsqueeze(1).unsqueeze(2).expand(-1, pred.size(1), 1))
                pred_t = pred_t.squeeze(2)
                
                # Calculate loss per sample and outcome
                loss_per_sample = F.mse_loss(pred_t, batch_y, reduction='none')  # [batch_size, num_outcomes]
                
                # Sum up total loss and count samples
                test_loss += loss_per_sample.sum().item()
                test_samples += loss_per_sample.numel()
        
        avg_train_loss = train_loss / train_batches
        avg_test_loss = test_loss / test_samples
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: train_loss = {avg_train_loss:.4f}, test_loss = {avg_test_loss:.4f}")
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        # Calculate MSE for each treatment-outcome combination
        num_outcomes = Y_test.shape[1]
        num_treatments = len(torch.unique(T))
        mse_matrix = torch.zeros(num_treatments, num_outcomes)
        
        data_dict = {"feature": X_test}
        all_preds = model(data_dict)  # [batch, num_outcomes, num_treatments]
        
        for t in range(num_treatments):
            mask = (T_test == t)
            if mask.any():
                t_pred = all_preds[mask, :, t]  # [batch_t, num_outcomes]
                t_true = Y_test[mask]
                t_mse = F.mse_loss(t_pred, t_true, reduction='none').mean(0)
                mse_matrix[t] = t_mse
                print(f"Treatment {t} samples: {mask.sum().item()}")
        
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

def evaluate_tlearner_all_scenarios():
    """Evaluate T-learner on all synthetic scenarios"""
    scenarios = generate_all_scenarios()
    results = {}
    
    scenario_desc = {
        'A': 'Multiple T, Multiple O, No Conf',
        'B': 'Multiple T, Multiple O, With Conf'
    }
    
    # Define scenario-specific hyperparameters
    scenario_params = {
        'A': {'num_epochs': 100},
        'B': {'num_epochs': 100},
    }
    
    print("\nEvaluating T-learner across all scenarios...")
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
            
            model = Tlearner(
                input_dim=input_dim,
                num_outcome=num_outcomes,
                num_treatment=num_treatments,
                task="regression"
            )
            
            detailed_results = train_and_evaluate_tlearner(
                model, 
                data,
                num_epochs=params['num_epochs'],
                batch_size=256
            )
            results[scenario_key] = detailed_results
            
            print(f"Overall MSE: {detailed_results['overall_mse']:.4f}")
            print(f"Average CATE RMSE: {detailed_results['cate_rmse']['average']:.4f}")
            
        except Exception as e:
            print(f"Error in scenario {scenario_key}: {str(e)}")
            results[scenario_key] = None
    
    return results

if __name__ == "__main__":
    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    results = evaluate_tlearner_all_scenarios()
    
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