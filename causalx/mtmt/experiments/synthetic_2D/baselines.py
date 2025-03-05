import sys
sys.path.insert(0, '../../')  # Keep this to find local modules
import numpy as np
import torch
import torch.nn.functional as F
from DGP import generate_all_scenarios
from catenets.models.jax import SNet1, DRNet
from model.model import PLE2D, PLE2D_DA
from util.utility import get_da_loss  # Import the DA loss class and get_da_loss function

def train_and_evaluate(model, data, num_epochs=100):
    """Basic training loop for models without domain adaptation"""
    X = torch.FloatTensor(data['X'])
    T = torch.LongTensor(data['T'])
    Y = torch.FloatTensor(data['Y'])
    
    # Split data
    n_train = int(0.8 * len(X))
    train_idx = np.random.choice(len(X), n_train, replace=False)
    test_idx = np.array(list(set(range(len(X))) - set(train_idx)))
    
    X_train, T_train, Y_train = X[train_idx], T[train_idx], Y[train_idx]
    X_test, T_test, Y_test = X[test_idx], T[test_idx], Y[test_idx]
    
    optimizer = torch.optim.Adam(model.parameters())
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        
        # Prepare input data dictionary
        data_dict = {"feature": X_train, "treatment": T_train}
        pred = model(data_dict)
        
        # MSE loss
        loss = F.mse_loss(pred, Y_train)
        loss.backward()
        optimizer.step()
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        test_dict = {"feature": X_test, "treatment": T_test}
        pred = model(test_dict)
        mse = F.mse_loss(pred, Y_test).item()
    
    return mse

def train_and_evaluate_with_da(model, data, num_epochs=100, lambda_da=0.1, da_method='mmd'):
    """
    Training loop for PLE2D+DA with domain adaptation loss using MMD or Wasserstein
    
    Args:
        model: The PLE2D_DA model
        data: Dictionary containing X, T, Y
        num_epochs: Number of training epochs
        lambda_da: Weight for domain adaptation loss
        da_method: 'mmd' or 'wasserstein' for domain adaptation
    """
    X = torch.FloatTensor(data['X'])
    T = torch.LongTensor(data['T'])
    Y = torch.FloatTensor(data['Y'])
    
    # Split data
    n_train = int(0.8 * len(X))
    train_idx = np.random.choice(len(X), n_train, replace=False)
    test_idx = np.array(list(set(range(len(X))) - set(train_idx)))
    
    X_train, T_train, Y_train = X[train_idx], T[train_idx], Y[train_idx]
    X_test, T_test, Y_test = X[test_idx], T[test_idx], Y[test_idx]
    
    optimizer = torch.optim.Adam(model.parameters())
    da_loss_fn = get_da_loss(method=da_method)  # Just get the loss function
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        
        # Forward pass with domain adaptation
        data_dict = {"feature": X_train, "treatment": T_train}
        pred, treatment_shared_outputs = model(data_dict)
        
        # Main prediction loss
        pred_loss = F.mse_loss(pred, Y_train)
        
        # Domain adaptation loss using MMD or Wasserstein
        da_loss = 0
        if treatment_shared_outputs is not None:
            # Group representations by treatment
            unique_treatments = torch.unique(T_train)
            treatment_groups = {t.item(): treatment_shared_outputs[T_train == t] 
                              for t in unique_treatments}
            
            # Compute pairwise DA loss between treatment groups
            n_treatments = len(treatment_groups)
            for i in range(n_treatments):
                for j in range(i + 1, n_treatments):
                    t1_reps = treatment_groups[i]
                    t2_reps = treatment_groups[j]
                    da_loss += da_loss_fn(t1_reps, t2_reps)
            
            # Normalize by number of comparisons
            n_comparisons = (n_treatments * (n_treatments - 1)) // 2
            if n_comparisons > 0:
                da_loss /= n_comparisons
        
        # Combined loss
        loss = pred_loss + lambda_da * da_loss
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:  # Print progress every 10 epochs
            print(f"Epoch {epoch}: pred_loss = {pred_loss:.4f}, da_loss = {da_loss:.4f}")
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        test_dict = {"feature": X_test, "treatment": T_test}
        pred, _ = model(test_dict)
        mse = F.mse_loss(pred, Y_test).item()
    
    return mse

def evaluate_binary_multiple_outcome(data):
    """
    Evaluate models specifically for binary treatment, multiple outcome scenario (Scenario A)
    Using T-learner as the primary baseline
    """
    X = torch.FloatTensor(data['X'])
    T = torch.LongTensor(data['T'])
    Y = torch.FloatTensor(data['Y'])
    
    results = {}
    
    # 1. T-learner (Primary Baseline)
    model_t = Tlearner(
        input_dim=X.shape[1],
        num_outcome=Y.shape[1],
        num_treatment=2,
        task="regression"
    )
    results['T-learner (Baseline)'] = train_and_evaluate(model_t, data)
    
    # 2. PLE2D
    model = PLE2D(
        input_dim=X.shape[1],
        num_outcomes=Y.shape[1],
        num_treatments=2,
        task="regression",
        num_layers=2,
        expert_hidden_units=[64, 32],
        num_shared_experts=2,
        num_specific_experts=2
    )
    results['PLE2D'] = train_and_evaluate(model, data)
    
    # 3. PLE2D+DA
    model_da = PLE2D_DA(
        input_dim=X.shape[1],
        num_outcomes=Y.shape[1],
        num_treatments=2,
        task="regression",
        num_layers=2,
        expert_hidden_units=[64, 32],
        num_shared_experts=2,
        num_specific_experts=2
    )
    results['PLE2D+DA'] = train_and_evaluate_with_da(model_da, data)
    
    # Optional: Add TARNet as another comparison
    model_tar = TARNet(
        input_dim=X.shape[1],
        num_outcome=Y.shape[1],
        num_treatment=2,
        task="regression"
    )
    results['TARNet'] = train_and_evaluate(model_tar, data)
    
    return results

def evaluate_all_baselines(da_method='mmd'):
    scenarios = generate_all_scenarios()
    results = {
        'T-learner (Baseline)': [],    # Primary baseline for binary treatment
        'TARNet': [],                  # Secondary baseline
        'PLE2D': [],                   # Our method without DA
        'PLE2D+DA (Ours)': []          # Our proposed method
    }
    
    # For scenario A (Binary T, Multiple O, No Conf), use specialized evaluation
    scenario_a_results = evaluate_binary_multiple_outcome(scenarios['A'])
    
    # Add scenario A results to the main results dictionary
    for method in results.keys():
        results[method].append(scenario_a_results[method])
    
    # Continue with other scenarios...
    for scenario in ['B', 'C', 'D', 'E', 'F']:
        data = scenarios[scenario]
        X = data['X']
        T = data['T']
        Y = data['Y']
        input_dim = X.shape[1]
        num_treatments = len(np.unique(T))
        num_outcomes = Y.shape[1]
        
        # 1. Single-Outcome (Binary T) - CATENets SNet1
        # For each treatment pair and outcome
        all_mses = []
        treatments = np.unique(T)
        for t1 in range(len(treatments)):
            for t2 in range(t1 + 1, len(treatments)):
                # Get indices for this treatment pair
                mask = (T == treatments[t1]) | (T == treatments[t2])
                X_pair = X[mask]
                T_pair = T[mask]
                # Convert to binary treatment
                T_pair = (T_pair == treatments[t2]).astype(int)
                Y_pair = Y[mask]
                
                # Train for each outcome
                for o in range(num_outcomes):
                    model = SNet1(binary_y=False)
                    model.fit(X=X_pair, y=Y_pair[:, o], w=T_pair)
                    preds = model.predict(X_pair)
                    mse = np.mean((preds - Y_pair[:, o])**2)
                    all_mses.append(mse)
        
        # Average MSE across all treatment pairs and outcomes
        avg_mse = np.mean(all_mses)
        results['Single-Outcome (Binary T)'].append(avg_mse)
        
        # 2. Separate (T,O)-Models - Our Tlearner
        model = Tlearner(
            input_dim=input_dim,
            num_outcome=num_outcomes,
            num_treatment=num_treatments,
            task="regression"
        )
        mse = train_and_evaluate(model, data)
        results['Separate (T,O)-Models'].append(mse)
        
        # 3. Naive Multi-Task - Our TARNet
        model = TARNet(
            input_dim=input_dim,
            num_outcome=num_outcomes,
            num_treatment=num_treatments,
            task="regression"
        )
        mse = train_and_evaluate(model, data)
        results['Naive Multi-Task'].append(mse)
        
        # 4. DRNet-like - CATENets DRNet
        # Similar approach for DRNet
        all_mses = []
        for t1 in range(len(treatments)):
            for t2 in range(t1 + 1, len(treatments)):
                mask = (T == treatments[t1]) | (T == treatments[t2])
                X_pair = X[mask]
                T_pair = T[mask]
                T_pair = (T_pair == treatments[t2]).astype(int)
                Y_pair = Y[mask]
                
                for o in range(num_outcomes):
                    drnet = DRNet(binary_y=False)
                    drnet.fit(X=X_pair, y=Y_pair[:, o], w=T_pair)
                    preds = drnet.predict(X_pair)
                    mse = np.mean((preds - Y_pair[:, o])**2)
                    all_mses.append(mse)
        
        avg_mse = np.mean(all_mses)
        results['DRNet-like'].append(avg_mse)
            
        # 5. PLE2D (without DA)
        model = PLE2D(
            input_dim=input_dim,
            num_outcomes=num_outcomes,
            num_treatments=num_treatments,
            task="regression",
            num_layers=2,
            expert_hidden_units=[64, 32],
            num_shared_experts=2,
            num_specific_experts=2
        )
        mse = train_and_evaluate(model, data)
        results['PLE2D'].append(mse)
        
        # 6. Our PLE2D+DA method
        model = PLE2D_DA(
            input_dim=input_dim,
            num_outcomes=num_outcomes,
            num_treatments=num_treatments,
            task="regression",
            num_layers=2,
            expert_hidden_units=[64, 32],
            num_shared_experts=2,
            num_specific_experts=2
        )
        mse = train_and_evaluate_with_da(model, data, da_method=da_method)
        results['PLE2D+DA (Ours)'].append(mse)
    
    return results

def analyze_results(results_mmd, results_wasserstein):
    """Analyze results across multiple dimensions"""
    # Update scenario descriptions
    scenarios = {
        'A': 'Binary T, Multiple O, No Conf',
        'B': 'Multiple T, Single O, No Conf',
        'C': 'Binary T, Multiple O, With Conf',
        'D': 'Multiple T, Single O, With Conf',
        'E': 'Multiple T, Multiple O, No Conf',
        'F': 'Multiple T, Multiple O, With Conf'
    }
    
    for results, method_name in [(results_mmd, "MMD"), (results_wasserstein, "Wasserstein")]:
        print(f"\n=== Analysis with {method_name} ===")
        
        # 1. Multiple Treatment Analysis
        print("\n1. Multiple Treatment Handling Analysis:")
        print("-" * 60)
        # Compare scenarios with binary vs multiple treatments
        binary_scenarios = ['A', 'C']  # Binary treatment scenarios
        multi_scenarios = ['B', 'D', 'E', 'F']  # Multiple treatment scenarios
        
        # Get indices for binary and multi treatment scenarios
        binary_idx = [list(scenarios.keys()).index(s) for s in binary_scenarios]
        multi_idx = [list(scenarios.keys()).index(s) for s in multi_scenarios]
        
        # Compare PLE2D+DA performance in binary vs multiple treatment scenarios
        binary_perf = np.mean([results['PLE2D+DA (Ours)'][i] for i in binary_idx])
        multi_perf = np.mean([results['PLE2D+DA (Ours)'][i] for i in multi_idx])
        
        print("PLE2D+DA Performance:")
        print(f"Binary treatment scenarios MSE: {binary_perf:.3f}")
        print(f"Multiple treatment scenarios MSE: {multi_perf:.3f}")
        
        # 2. Multiple Outcome Analysis
        print("\n2. Multiple Outcome Handling Analysis:")
        print("-" * 60)
        single_o_scenarios = ['B', 'D']  # Single outcome scenarios
        multi_o_scenarios = ['A', 'C', 'E', 'F']  # Multiple outcome scenarios
        
        single_o_idx = [list(scenarios.keys()).index(s) for s in single_o_scenarios]
        multi_o_idx = [list(scenarios.keys()).index(s) for s in multi_o_scenarios]
        
        single_o_perf = np.mean([results['PLE2D+DA (Ours)'][i] for i in single_o_idx])
        multi_o_perf = np.mean([results['PLE2D+DA (Ours)'][i] for i in multi_o_idx])
        
        print("PLE2D+DA Performance:")
        print(f"Single outcome scenarios MSE: {single_o_perf:.3f}")
        print(f"Multiple outcome scenarios MSE: {multi_o_perf:.3f}")
        
        # 3. Confounding Analysis
        print("\n3. Confounding Scenarios Analysis:")
        print("-" * 60)
        conf_scenarios = ['C', 'D', 'F']  # Scenarios with confounding
        no_conf_scenarios = ['A', 'B', 'E']  # Scenarios without confounding
        
        conf_idx = [list(scenarios.keys()).index(s) for s in conf_scenarios]
        no_conf_idx = [list(scenarios.keys()).index(s) for s in no_conf_scenarios]
        
        # Compare PLE2D vs PLE2D+DA in confounded scenarios
        ple2d_conf = np.mean([results['PLE2D'][i] for i in conf_idx])
        ple2d_da_conf = np.mean([results['PLE2D+DA (Ours)'][i] for i in conf_idx])
        conf_improvement = ((ple2d_conf - ple2d_da_conf) / ple2d_conf) * 100
        
        print("Confounded Scenarios (C, D, F):")
        print(f"PLE2D MSE: {ple2d_conf:.3f}")
        print(f"PLE2D+DA MSE: {ple2d_da_conf:.3f}")
        print(f"Improvement: {conf_improvement:.1f}%")
        
        # Print scenario-wise analysis
        print("\nDetailed Scenario Analysis:")
        print("-" * 60)
        print("Scenario Description                  PLE2D    PLE2D+DA    Improvement")
        for scenario, desc in scenarios.items():
            idx = list(scenarios.keys()).index(scenario)
            ple2d_mse = results['PLE2D'][idx]
            ple2d_da_mse = results['PLE2D+DA (Ours)'][idx]
            improvement = ((ple2d_mse - ple2d_da_mse) / ple2d_mse) * 100
            print(f"{scenario}: {desc:<30} {ple2d_mse:.3f}    {ple2d_da_mse:.3f}    {improvement:>6.1f}%")

if __name__ == "__main__":
    # Test with both MMD and Wasserstein
    print("\nRunning experiments...")
    results_mmd = evaluate_all_baselines(da_method='mmd')
    results_wasserstein = evaluate_all_baselines(da_method='wasserstein')
    
    # Print detailed results table
    scenarios = ['A', 'B', 'C', 'D', 'E', 'F']
    for results, method_name in [(results_mmd, "MMD"), (results_wasserstein, "Wasserstein")]:
        print(f"\nMean Squared Error (MSE) Results with {method_name}:")
        print("-" * 70)
        print("Method                    A     B     C     D     E     F")
        print("-" * 70)
        
        for method, mses in results.items():
            row = f"{method:<20} "
            for mse in mses:
                if mse is None:
                    row += "  --  "
                else:
                    row += f" {mse:.2f} "
            print(row)
    
    # Run detailed analysis
    analyze_results(results_mmd, results_wasserstein) 