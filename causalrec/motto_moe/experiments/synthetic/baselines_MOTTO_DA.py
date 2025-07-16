# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict
import sys
sys.path.insert(0, '../../')
import numpy as np
import torch
import torch.nn.functional as F
from DGP import generate_all_scenarios
from model.model import MOTTO_DA
from geomloss import SamplesLoss  # Import geomloss

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def print_gpu_memory():
    if torch.cuda.is_available():
        print(f"GPU Memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"GPU Memory cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

def train_and_evaluate_motto_da(model, data, num_epochs=60, batch_size=256, da_weight=1.0, loss_th=1000.0):
    """Training loop for MOTTO_DA with Distribution Alignment using Wasserstein/MMD distance"""
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    
    # Initialize the distribution distance loss
    # Use "gaussian" for MMD or "sinkhorn" for Wasserstein
    da_loss = SamplesLoss(loss="energy", backend="tensorized")
    
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
    
    # Create DataLoader
    train_dataset = torch.utils.data.TensorDataset(X_train, T_train, Y_train)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True
    )
    
    optimizer = torch.optim.Adam(model.parameters(),lr=0.002)
    
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
            pred, treatment_shared_outputs = model(data_dict)
            
            # Get predictions for actual treatments
            pred_t = torch.gather(pred, 2, batch_t.unsqueeze(1).unsqueeze(2).expand(-1, pred.size(1), 1))
            pred_t = pred_t.squeeze(2)  # [batch, num_outcomes]
            
            # Prediction loss
            pred_loss = F.mse_loss(pred_t, batch_y)
            
            # Distribution alignment loss
            num_experts = treatment_shared_outputs.shape[0]
            batch_size = pred.shape[0]
            da_losses = []
            
            # Process each expert independently
            for expert_idx in range(num_experts):
                expert_outputs = treatment_shared_outputs[expert_idx]  # [batch, hidden_dim]
                
                # Group outputs by treatment for this expert
                treatment_groups = {}
                for i in range(len(batch_t)):
                    t = batch_t[i].item()
                    if t not in treatment_groups:
                        treatment_groups[t] = []
                    treatment_groups[t].append(expert_outputs[i])
                
                # Skip if we don't have enough samples for comparison
                if len(treatment_groups) < 2:
                    continue
                
                # Combine all data for this expert
                all_data = torch.cat([torch.stack(treatment_groups[t]) for t in treatment_groups], dim=0)
                
                # Compare each treatment group against the combined data
                expert_loss = 0
                num_comparisons = 0
                for t in treatment_groups:
                    t_data = torch.stack(treatment_groups[t])
                    # Use geomloss instead of custom MMD
                    expert_loss += da_loss(t_data, all_data) / batch_size
                    num_comparisons += 1
                
                if num_comparisons > 0:
                    da_losses.append(expert_loss / num_comparisons)
            
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
        print("
Detailed MSE Analysis:")
        print("Treatment | " + " | ".join([f"Outcome {i}" for i in range(num_outcomes)]))
        print("-" * (10 + 12 * num_outcomes))
        for t in range(num_treatments):
            mse_values = [f"{mse:.4f}" for mse in mse_matrix[t]]
            print(f"    {t}    | " + " | ".join(mse_values))
        
        print("
CATE RMSE Analysis:")
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

def evaluate_motto_da_all_scenarios():
    """Evaluate MOTTO_DA on all synthetic scenarios"""
    scenarios = generate_all_scenarios()
    results = {}
    
    scenario_desc = {
        'A': 'Multiple T, Multiple O, No Conf',
        'B': 'Multiple T, Multiple O, With Conf'
    }
    
    # Define scenario-specific hyperparameters
    scenario_params = {
        'A': {'da_weight': 1, 'num_epochs': 120},    # Multiple outcomes + treatments
        'B': {'da_weight': 10, 'num_epochs': 200}     # Most complex case
    }
    
    print("
Evaluating MOTTO_DA across all scenarios...")
    print("-" * 60)
    
    for scenario_key, data in scenarios.items():
        try:
            print(f"
Before scenario {scenario_key} - GPU Memory Status:")
            print_gpu_memory()
            
            # Reset random seeds for each scenario
            RANDOM_SEED = 42
            np.random.seed(RANDOM_SEED)
            torch.manual_seed(RANDOM_SEED)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(RANDOM_SEED)
                torch.cuda.manual_seed_all(RANDOM_SEED)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                torch.cuda.empty_cache()
            
            print(f"
After GPU cache clear - GPU Memory Status:")
            print_gpu_memory()
            
            params = scenario_params[scenario_key]
            print(f"
Scenario {scenario_key}: {scenario_desc[scenario_key]}")
            print(f"Using DA weight: {params['da_weight']}, Epochs: {params['num_epochs']}")
            
            X = data['X']
            T = data['T']
            Y = data['Y']
            
            input_dim = X.shape[1]
            num_treatments = len(np.unique(T))
            num_outcomes = Y.shape[1] if len(Y.shape) > 1 else 1
            
            print(f"Data dimensions: Features={input_dim}, Treatments={num_treatments}, Outcomes={num_outcomes}")
            
            model = MOTTO_DA(
                num_layers=1,
                expert_type="mlp",
                num_shared_experts=1,
                num_outcome_shared_experts=1,
                num_treatment_shared_experts=1,
                num_specific_experts=0,
                num_outcomes=num_outcomes,
                num_treatments=num_treatments,
                input_hidden_units= [64],
                expert_hidden_units=[16,8],
                gate_hidden_units= [4], #[128, 64],
                tower_hidden_units=[16],
                hidden_activations="ReLU",
                net_dropout=0,
                batch_norm=False,
                input_dim=input_dim,
                task="regression"
            )
            
            detailed_results = train_and_evaluate_motto_da(
                model, 
                data,
                da_weight=params['da_weight'],
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
    results = evaluate_motto_da_all_scenarios()
    
    # Print summary table
    print("
Summary of Results")
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
