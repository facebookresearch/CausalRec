import sys
sys.path.insert(0, '../../')
import numpy as np
import torch
import torch.nn.functional as F
from DGP import generate_all_scenarios
from model_synthetic.TARNet import TARNet

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def calculate_cate_rmse(model, X, true_cate, test_idx):
    """
    Calculate RMSE of CATE predictions compared to true CATE.
    
    Args:
        model: Trained TARNet model
        X: Feature tensor
        true_cate: Dictionary containing true CATE values for each outcome
        test_idx: Indices for test set
    
    Returns:
        Dictionary containing RMSE for each outcome's CATE per treatment
    """
    model.eval()
    with torch.no_grad():
        # Get counterfactual predictions
        cf_preds = model.predict_counterfactuals(X)  # [N, num_treatment, num_outcome]
        
        # Calculate predicted CATE (comparing each treatment to control)
        pred_cate = cf_preds[:, 1:, :] - cf_preds[:, 0:1, :]  # [N, K-1, num_outcome]
        
        # Calculate RMSE for each outcome and treatment
        rmse_dict = {}
        num_treatments = pred_cate.shape[1]  # K-1 treatments (excluding control)
        num_outcomes = cf_preds.shape[-1]
        
        # Initialize RMSE matrix [num_outcomes, num_treatments]
        rmse_matrix = torch.zeros(num_outcomes, num_treatments)
        
        for m in range(num_outcomes):
            true_cate_m = torch.FloatTensor(true_cate[f'outcome{m}'])[test_idx]  # [N_test, K-1]
            pred_cate_m = pred_cate[:, :, m]  # [N_test, K-1]
            
            # Calculate RMSE for each treatment
            for k in range(num_treatments):
                rmse = torch.sqrt(((true_cate_m[:, k] - pred_cate_m[:, k]) ** 2).mean())
                rmse_matrix[m, k] = rmse
                rmse_dict[f'outcome{m}_treatment{k+1}'] = rmse.item()
        
        # Calculate average RMSE across all treatment-outcome pairs
        rmse_dict['average'] = rmse_matrix.mean().item()
        # Add average RMSE per outcome
        for m in range(num_outcomes):
            rmse_dict[f'outcome{m}_average'] = rmse_matrix[m].mean().item()
        # Add average RMSE per treatment
        for k in range(num_treatments):
            rmse_dict[f'treatment{k+1}_average'] = rmse_matrix[:, k].mean().item()
        
    return rmse_dict

def train_and_evaluate_tarnet(model, data, num_epochs=200, mmd_weight=0, batch_size=128):
    """Training loop for TARNet with MMD regularization and CATE evaluation"""
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    
    X = torch.FloatTensor(data['X'])
    T = torch.LongTensor(data['T'])
    Y = torch.FloatTensor(data['Y'])
    
    if len(Y.shape) == 1:
        Y = Y.unsqueeze(-1)
    
    # Split data
    n_train = int(0.8 * len(X))
    train_idx = np.random.choice(len(X), n_train, replace=False)
    test_idx = np.array(list(set(range(len(X))) - set(train_idx)))
    
    X_train, T_train, Y_train = X[train_idx], T[train_idx], Y[train_idx]
    X_test, T_test, Y_test = X[test_idx], T[test_idx], Y[test_idx]
    
    # Create DataLoader for batched training
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
        num_batches = 0
        
        for batch_x, batch_t, batch_y in train_loader:
            optimizer.zero_grad()
            
            # Get shared representations for batch
            shared_reprs = model.get_representations(batch_x)
            
            # Forward pass
            data_dict = {"feature": batch_x, "treatment": batch_t}
            pred = model(data_dict)
            
            # Compute prediction loss
            pred_loss = F.mse_loss(pred, batch_y)
            
            # Compute MMD loss between treatment groups for each outcome
            mmd_losses = model.compute_all_mmd_losses(batch_x, batch_t)
            total_mmd_loss = sum(mmd_losses) / len(mmd_losses)  # Average MMD loss across outcomes
            
            # Total loss
            total_loss = pred_loss + mmd_weight * total_mmd_loss
            #print(f"Epoch {epoch}: pred_loss = {pred_loss:.4f}, mmd_loss = {total_mmd_loss:.4f}, total_loss = {total_loss:.4f}")
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += total_loss.item()
            num_batches += 1
        
        avg_epoch_loss = epoch_loss / num_batches
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: avg_loss = {avg_epoch_loss:.4f}")
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        test_dict = {"feature": X_test, "treatment": T_test}
        pred = model(test_dict)
        
        # Calculate MSE for each treatment-outcome combination
        unique_treatments = torch.unique(T_test)
        num_outcomes = Y_test.shape[1]
        mse_matrix = torch.zeros(len(unique_treatments), num_outcomes)
        
        for t_idx, t in enumerate(unique_treatments):
            mask = (T_test == t)
            if mask.any():
                t_pred = pred[mask]
                t_true = Y_test[mask]
                t_mse = F.mse_loss(t_pred, t_true, reduction='none').mean(0)
                mse_matrix[t_idx] = t_mse
        
        # Calculate CATE RMSE using test set indices
        cate_rmse = calculate_cate_rmse(model, X_test, data['true_cate'], test_idx)
        
        # Print results
        print("\nDetailed MSE Analysis:")
        print("MSE for each treatment-outcome combination:")
        print("Treatment | " + " | ".join([f"Outcome {i}" for i in range(num_outcomes)]))
        print("-" * (10 + 12 * num_outcomes))
        for t_idx, t in enumerate(unique_treatments):
            mse_values = [f"{mse:.4f}" for mse in mse_matrix[t_idx]]
            print(f"    {t}    | " + " | ".join(mse_values))
        
        print("\nCATE RMSE Analysis:")
        # Print RMSE matrix
        print("\nCATE RMSE for each treatment-outcome pair:")
        print("         | " + " | ".join([f"Treatment {k+1}" for k in range(len(unique_treatments)-1)]) + " | Average")
        print("-" * (10 + 14 * (len(unique_treatments))))
        
        for m in range(num_outcomes):
            rmse_values = [f"{cate_rmse[f'outcome{m}_treatment{k+1}']:.4f}" for k in range(len(unique_treatments)-1)]
            rmse_values.append(f"{cate_rmse[f'outcome{m}_average']:.4f}")
            print(f"Outcome {m} | " + " | ".join(rmse_values))
        
        # Print treatment averages
        treatment_avgs = [f"{cate_rmse[f'treatment{k+1}_average']:.4f}" for k in range(len(unique_treatments)-1)]
        treatment_avgs.append(f"{cate_rmse['average']:.4f}")
        print("-" * (10 + 14 * (len(unique_treatments))))
        print(f"Average  | " + " | ".join(treatment_avgs))
        
        detailed_results = {
            'overall_mse': mse_matrix.mean().item(),
            'mse_matrix': mse_matrix.tolist(),
            'num_treatments': len(unique_treatments),
            'num_outcomes': num_outcomes,
            'cate_rmse': cate_rmse
        }
    
    return detailed_results

def evaluate_tarnet_all_scenarios():
    """Evaluate TARNet on all synthetic scenarios"""
    scenarios = generate_all_scenarios()
    results = {}
    
    scenario_desc = {
        'A': 'Binary T, Multiple O, No Conf',
        'B': 'Multiple T, Single O, No Conf',
        'C': 'Binary T, Multiple O, With Conf',
        'D': 'Multiple T, Single O, With Conf',
        'E': 'Multiple T, Multiple O, No Conf',
        'F': 'Multiple T, Multiple O, With Conf'
    }
    
    # Define MMD weights for different scenarios
    mmd_weights = {
        'A': 1,   
        'B': 1,  
        'C': 20,   
        'D': 20,   
        'E': 1,  
        'F': 15   
    }
    
    print("\nEvaluating TARNet across all scenarios...")
    print("-" * 60)
    
    for scenario_key, data in scenarios.items():
        try:
            print(f"\nScenario {scenario_key}: {scenario_desc[scenario_key]}")
            print(f"Using MMD weight: {mmd_weights[scenario_key]}")
            
            X = data['X']
            T = data['T']
            Y = data['Y']
            
            input_dim = X.shape[1]
            num_treatments = len(np.unique(T))
            num_outcomes = Y.shape[1] if len(Y.shape) > 1 else 1
            
            print(f"Data dimensions: Features={input_dim}, Treatments={num_treatments}, Outcomes={num_outcomes}")
            
            model = TARNet(
                input_dim=input_dim,
                num_outcome=num_outcomes,
                num_treatment=num_treatments,
                hidden_dims=[16,8],#[64, 32],
                task="regression"
            )
            
            detailed_results = train_and_evaluate_tarnet(
                model, 
                data,
                mmd_weight=mmd_weights[scenario_key],
                num_epochs=60,
                batch_size=256
            )
            results[scenario_key] = detailed_results
            print(f"Overall Prediction MSE: {detailed_results['overall_mse']:.4f}")
            print(f"Average CATE RMSE: {detailed_results['cate_rmse']['average']:.4f}")
            
        except Exception as e:
            print(f"Error in scenario {scenario_key}: {str(e)}")
            results[scenario_key] = None
    
    return results

if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(RANDOM_SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    results = evaluate_tarnet_all_scenarios()
    
    # Print summary table
    print("\nSummary of Results")
    print("-" * 80)
    print("Scenario    Description                          Pred MSE    CATE RMSE")
    print("-" * 80)
    for scenario, res in results.items():
        if res is not None:
            desc = {
                'A': 'Binary T, Multiple O, No Conf',
                'B': 'Multiple T, Single O, No Conf',
                'C': 'Binary T, Multiple O, With Conf',
                'D': 'Multiple T, Single O, With Conf',
                'E': 'Multiple T, Multiple O, No Conf',
                'F': 'Multiple T, Multiple O, With Conf'
            }
            print(f"{scenario:<10} {desc[scenario]:<35} {res['overall_mse']:.4f}      {res['cate_rmse']['average']:.4f}") 