import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.stats import f_oneway
import yaml

############################
# 1. Data Generation Components
############################

class SmallMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=16):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        h = F.relu(self.fc1(x))
        out = self.fc2(h)
        return out.squeeze(-1)

def generate_correlated_features(n_samples, n_features, Sigma_X=None):
    """Generate features with specified covariance structure and nonlinear transformations
    using a neural network.
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate
    n_features : int
        Number of base features to generate
    Sigma_X : array-like, optional
        Covariance matrix for base features. If None, uses identity.
        
    Returns:
    --------
    X : ndarray
        Array of shape (n_samples, n_features) containing the generated features
    """
    # Generate base features with larger variance
    if Sigma_X is None:
        X_base = 3.0 * np.random.randn(n_samples, n_features)  # Increased scale from 1.0 to 3.0
    else:
        # Scale up the covariance matrix
        scaled_Sigma_X = 9.0 * Sigma_X  # Increased variance by factor of 9
        X_base = np.random.multivariate_normal(
            mean=np.zeros(n_features),
            cov=scaled_Sigma_X,
            size=n_samples
        )
    
    # Create and initialize a simple MLP for feature transformation
    class FeatureTransformer(nn.Module):
        def __init__(self, input_dim, hidden_dim=64):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, input_dim)
            
        def forward(self, x):
            h = torch.tanh(self.fc1(x))
            return self.fc2(h)
    
    # Initialize and apply the transformer
    transformer = FeatureTransformer(n_features)
    X_torch = torch.FloatTensor(X_base)
    with torch.no_grad():
        X = 5.0 * transformer(X_torch).numpy()  # Keep original scale of 5.0

    return X, X_base

class TreatmentMLP(nn.Module):
    def __init__(self, input_dim, num_treatments, hidden_dim=32):
        super().__init__()
        self.fc1 = nn.Linear(input_dim + num_treatments, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        
    def forward(self, x, t_onehot):
        inputs = torch.cat([x, t_onehot], dim=1)
        h = F.relu(self.fc1(inputs))
        out = torch.tanh(self.fc2(h))
        out = 10*out
        return out.squeeze(-1)

def generate_data(
    N=10000, 
    d=10,
    num_treatments=3,
    num_outcomes=2,  # Can be 1 or 2
    confounding_level=10.0,
    outcome_correlation=0.5,
    seed=42,
    verbose=False
):
    """
    Generate synthetic data according to the specified DGP.
    
    Parameters:
    -----------
    N : int
        Number of samples
    d : int
        Number of features
    num_treatments : int
        Number of treatments
    num_outcomes : int
        Number of outcomes (1 or 2)
    confounding_level : float
        Strength of confounding
    outcome_correlation : float
        Correlation between outcomes (p in [-1,1]), only used if num_outcomes=2
    noise_scale : float
        Scale of noise terms
    seed : int
        Random seed
    verbose : bool
        Whether to print additional information during data generation
    """
    assert num_outcomes in [1, 2], "num_outcomes must be 1 or 2"
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # 1. Generate covariates
    X, X_base = generate_correlated_features(N, d)
    X_first5 = X[:, :5]
    
    beta_0 = np.random.normal(0, 0.01, size=N)    
    beta_k = np.random.normal(0, 1.0, size=(num_treatments, d))
    
    alpha_list = []
    temperature = 0.5
    for k in range(num_treatments):
        if confounding_level > 0:
            base_effect = X_base.dot(beta_k[k])
            # Stronger bias term that scales with treatment index and confounding
            bias = -k * confounding_level * temperature  # More direct control over treatment assignment
            alpha_k = beta_0 + base_effect + bias
        else:
            alpha_k = beta_0
        alpha_list.append(alpha_k)
    alpha_np = np.vstack(alpha_list).T
    
    # Use fixed temperature with softmax
    alpha_torch = torch.FloatTensor(alpha_np)/temperature
    propensity = nn.Softmax(dim=1)(alpha_torch).numpy()
    
    T = np.array([np.random.choice(num_treatments, p=p) for p in propensity])
    T_onehot = np.zeros((N, num_treatments))
    T_onehot[np.arange(N), T] = 1
    

    # Print treatment counts
    unique, counts = np.unique(T, return_counts=True)
    if verbose:
        print("Treatment Counts:")
        for t, c in zip(unique, counts):
            print(f"T={t}: {c} ({c/N*100:.1f}%)")
    

    # 3. Generate orthonormal basis vectors
    u1 = np.random.normal(0, 1, size=5)
    u1 = u1 / np.linalg.norm(u1)
    
    if num_outcomes == 2:
        u2 = np.random.normal(0, 1, size=5)
        u2 = u2 - (u2 @ u1) * u1  # Make orthogonal to u1
        u2 = u2 / np.linalg.norm(u2)
    
    # Generate weight vectors
    c = 2.0  # Scaling constant
    w1 = c * u1
    if num_outcomes == 2:
        w2 = c * (np.sqrt(outcome_correlation) * u1 + np.sqrt(1 - outcome_correlation) * u2)
        w_vectors = [w1, w2]
    else:
        w_vectors = [w1]
    
    # 4. Setup MLPs for nonlinear terms
    mlp1 = TreatmentMLP(d, num_treatments)
    mlps = [mlp1]
    if num_outcomes == 2:
        mlp2 = TreatmentMLP(d, num_treatments)
        
        # Strengthen correlation between MLPs
        mix_ratio = np.sqrt(outcome_correlation)
        
        with torch.no_grad():
            new_fc1_weight = mix_ratio * mlp1.fc1.weight + (1 - mix_ratio) * mlp2.fc1.weight
            new_fc1_bias = mix_ratio * mlp1.fc1.bias + (1 - mix_ratio) * mlp2.fc1.bias
            mlp2.fc1.weight.copy_(new_fc1_weight)
            mlp2.fc1.bias.copy_(new_fc1_bias)
            
            # Correlate fc2 weights
            new_fc2_weight = mix_ratio * mlp1.fc2.weight + (1 - mix_ratio) * mlp2.fc2.weight
            new_fc2_bias = mix_ratio * mlp1.fc2.bias + (1 - mix_ratio) * mlp2.fc2.bias
            mlp2.fc2.weight.copy_(new_fc2_weight)
            mlp2.fc2.bias.copy_(new_fc2_bias)
            
            # Add additional correlation in activation patterns
            mlp2.fc1.weight.data *= (1 + 0.2 * torch.randn_like(mlp2.fc1.weight))
            mlp2.fc2.weight.data *= (1 + 0.2 * torch.randn_like(mlp2.fc2.weight))
        
        mlps.append(mlp2)
    
    # 5. Generate outcomes
    Y = np.zeros((N, num_outcomes))
    true_cate = {f'outcome{m}': np.zeros((N, num_treatments-1)) 
                 for m in range(num_outcomes)}
    
    # Parameters for outcomes - adjust scales for multiple treatments
    gamma_0 = np.random.normal(0, 0.1, size=num_outcomes)
    gamma_m = np.random.normal(5, 1, size=(num_outcomes, d))
    delta_m = np.random.normal(5, 0.5, size=(num_outcomes, num_treatments))  # Increased std from 0.1
    alpha_0 = np.random.normal(0, 0.5, size=num_outcomes)  # Increased std from 0.1
    alpha_t = np.random.normal(10, 0.5, size=(num_outcomes, num_treatments))  # Increased std from 0.1
    
    # Convert inputs to torch tensors for MLPs
    X_torch = torch.FloatTensor(X)
    T_onehot_torch = torch.FloatTensor(T_onehot)
    
    for m in range(num_outcomes):
        w_m = w_vectors[m]
        
        # Baseline effects
        B_m = (gamma_0[m] + 
               X.dot(gamma_m[m]) + 
               alpha_0[m] * X_first5.dot(w_m))
        
        # Treatment effects
        E_m = np.zeros(N)
        phi_m = mlps[m]
        
        for t in range(num_treatments):
            mask = (T == t)
            if mask.any():
                phi_mt = phi_m(X_torch[mask], T_onehot_torch[mask]).detach().numpy()
                E_m[mask] = (delta_m[m, t] + 
                            alpha_t[m, t] * X_first5[mask].dot(w_m) + phi_mt)
        
        Y[:, m] = B_m + E_m
        
        # Calculate true CATE
        for t in range(1, num_treatments):
            # Convert treatment one-hot to proper batch dimension
            t_onehot = torch.eye(num_treatments)[t].expand(N, -1)
            t0_onehot = torch.eye(num_treatments)[0].expand(N, -1)

            phi_t = phi_m(X_torch, t_onehot).detach().numpy()
            phi_0 = phi_m(X_torch, t0_onehot).detach().numpy()
            
            true_cate[f'outcome{m}'][:, t-1] = (
                delta_m[m, t] - delta_m[m, 0] +
                (alpha_t[m, t] - alpha_t[m, 0]) * X_first5.dot(w_m) +
                (phi_t - phi_0)
            )
    
    # Add noise
    noise = np.random.randn(N, 1)
    Y += noise
    
    if verbose:
        analyze_feature_treatment_relationships(X, T)
    
    return {
        'X': X,
        'T': T,
        'Y': Y,
        'true_cate': true_cate,
        'propensity': propensity,
        'w_vectors': w_vectors,
        'mlps': mlps
    }

def analyze_component_correlations(M=2, N=3000):
    """Analyze correlations between different components of the outcomes."""
    p_values = np.linspace(0, 1, 11)
    outcome_pairs = [(i,j) for i in range(M) for j in range(i+1,M)]
    
    results = {(i,j): {
        'weight_sims': [], 
        'psi_corrs': [], 
        'outcome_corrs': []
    } for i,j in outcome_pairs}
    
    for p in p_values:
        data = generate_data(N=N, outcome_correlation=p, num_outcomes=M, confounding_level=5.0, verbose=False)
        X_torch = torch.FloatTensor(data['X'])
        
        for i, j in outcome_pairs:
            # Weight vector similarity
            w_i, w_j = data['w_vectors'][i], data['w_vectors'][j]
            weight_sim = w_i.dot(w_j) / (np.linalg.norm(w_i) * np.linalg.norm(w_j))
            
            # Get baseline nonlinear components (without treatment)
            zeros = torch.zeros(N, data['T'].max() + 1)  # zero treatment encoding
            phi_i = data['mlps'][i](X_torch, zeros).detach().numpy()
            phi_j = data['mlps'][j](X_torch, zeros).detach().numpy()
            psi_corr = np.corrcoef(phi_i, phi_j)[0,1]
            
            # Final outcome correlation
            outcome_corr = np.corrcoef(data['Y'][:,i], data['Y'][:,j])[0,1]
            
            results[(i,j)]['weight_sims'].append(weight_sim)
            results[(i,j)]['psi_corrs'].append(psi_corr)
            results[(i,j)]['outcome_corrs'].append(outcome_corr)
    
    return results, p_values

def plot_component_correlations(results, p_values):
    """
    Plot correlations between different components of the outcomes.
    
    Parameters:
    -----------
    results : dict
        Results from analyze_component_correlations
    p_values : array
        Array of correlation parameters used
    """
    outcome_pairs = list(results.keys())
    plt.figure(figsize=(15, 5*len(outcome_pairs)))
    
    for idx, (i,j) in enumerate(outcome_pairs):
        # Weight similarities
        plt.subplot(len(outcome_pairs), 3, 3*idx + 1)
        plt.plot(p_values, results[(i,j)]['weight_sims'], 'o-')
        plt.xlabel('Correlation Parameter (p)')
        plt.ylabel(f'Weight Similarity ({i},{j})')
        plt.title(f'Weight Vector Similarity\nOutcomes {i} and {j}')
        plt.grid(True)
        
        # Nonlinear correlations
        plt.subplot(len(outcome_pairs), 3, 3*idx + 2)
        plt.plot(p_values, results[(i,j)]['psi_corrs'], 'o-')
        plt.xlabel('Correlation Parameter (p)')
        plt.ylabel(f'Psi Correlation ({i},{j})')
        plt.title(f'Nonlinear Component Correlation\nOutcomes {i} and {j}')
        plt.grid(True)
        
        # Outcome correlations
        plt.subplot(len(outcome_pairs), 3, 3*idx + 3)
        plt.plot(p_values, results[(i,j)]['outcome_corrs'], 'o-')
        plt.xlabel('Correlation Parameter (p)')
        plt.ylabel(f'Outcome Correlation ({i},{j})')
        plt.title(f'Final Outcome Correlation\nOutcomes {i} and {j}')
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def experiment_correlation_vs_cosine(M=2):
    """
    Experiment to analyze relationship between outcome correlation and weight vector similarity.
    
    Parameters:
    -----------
    M : int
        Number of outcomes to analyze
    """
    results, p_values = analyze_component_correlations(M=M, N=3000)
    plot_component_correlations(results, p_values)

def generate_all_scenarios(N=10000, d=10, num_outcomes=2, num_treatments=3,
                         confounding_level=5.0, outcome_correlation=0.1, 
                         base_seed=42):
    """Generate data for all six scenarios.
    
    Parameters:
    -----------
    N : int
        Number of samples
    d : int
        Number of features
    num_outcomes : int
        Number of outcomes
    num_treatments : int
        Number of treatments
    confounding_level : float
        Strength of confounding
    outcome_correlation : float
        Correlation between outcomes
    base_seed : int
        Base random seed
    """
    scenarios = {}
    
    # Define parameter modifications for each scenario
    scenario_params = {
        'A': dict(num_outcomes=2, num_treatments=3, confounding_level=0.0),  # Multiple T, Multiple O, No Conf
        'B': dict(num_outcomes=2, num_treatments=3, confounding_level=3.0),    # Multiple T, Multiple O, With Conf                                      # Multiple T, Multiple O, With Conf
    }
    
    # Generate data for each scenario
    base_params = {
        'N': N,
        'd': d,
        'num_treatments': num_treatments,
        'num_outcomes': num_outcomes,
        'confounding_level': confounding_level,
        'outcome_correlation': outcome_correlation
    }
    
    for scenario, override_params in scenario_params.items():
        # Create scenario-specific parameters by updating base parameters
        scenario_params = base_params.copy()
        scenario_params.update(override_params)
        scenario_params['seed'] = base_seed
        
        scenarios[scenario] = generate_data(**scenario_params)

    return scenarios

def analyze_feature_treatment_relationships(X, T, num_features_to_plot=5):
    """Analyze and visualize how features vary across treatments.
    
    Parameters:
    -----------
    X : ndarray
        Feature matrix of shape (n_samples, n_features)
    T : ndarray
        Treatment assignments of shape (n_samples,)
    num_features_to_plot : int
        Number of features to show in visualization
    """
    unique_treatments = np.unique(T)
    n_features = X.shape[1]
    num_treatments = len(unique_treatments)
    
    # Compute feature statistics by treatment
    treatment_means = []
    treatment_stds = []
    for t in unique_treatments:
        mask = (T == t)
        treatment_means.append(X[mask].mean(axis=0))
        treatment_stds.append(X[mask].std(axis=0))
    
    # Compute feature variation across treatments
    treatment_means = np.array(treatment_means)
    mean_differences = np.max(treatment_means, axis=0) - np.min(treatment_means, axis=0)
    
    # Find most varying features
    most_varying_idx = np.argsort(mean_differences)[-num_features_to_plot:]
    
    # Plot distributions
    plt.figure(figsize=(15, 10))
    
    # Treatment-specific boxplots for each selected feature
    plt.subplot(2, 1, 1)
    
    # Prepare data for boxplot
    boxplot_data = []
    labels = []
    for idx in most_varying_idx:
        for t in unique_treatments:
            mask = (T == t)
            boxplot_data.append(X[mask, idx])
            labels.append(f'X{idx+1}\nT={t}')
    
    # Create boxplot
    bp = plt.boxplot(boxplot_data, patch_artist=True)
    
    # Color boxes by treatment
    colors = plt.cm.Set3(np.linspace(0, 1, num_treatments))
    for i, box in enumerate(bp['boxes']):
        box.set(facecolor=colors[i % num_treatments])
    
    plt.xticks(range(1, len(labels) + 1), labels, rotation=45)
    plt.title('Feature Distributions by Treatment')
    plt.ylabel('Feature Value')
    
    # Mean differences plot
    plt.subplot(2, 1, 2)
    plt.bar(range(n_features), mean_differences)
    plt.xlabel('Feature Index')
    plt.ylabel('Max Mean Difference Across Treatments')
    plt.title('Feature Variation Across Treatments')
    
    plt.tight_layout()
    plt.show()
    
   

def generate_data_from_config(config):
    """
    Generate synthetic data according to configuration parameters.
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary containing data generation parameters
    """
    return generate_data(
        N=config['num_samples'],
        d=config['num_features'],
        num_treatments=config['num_treatments'],
        num_outcomes=config['num_outcomes'],
        confounding_level=config['confounding_level'],
        outcome_correlation=config['outcome_correlation'],
        seed=config['seed']
    )

def generate_scenarios_from_config(config):
    """
    Generate data for all scenarios defined in the config.
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary containing scenario definitions
    """
    scenarios = {}
    
    for scenario_name, scenario_params in config['scenarios'].items():
        # Create scenario-specific config by updating base config
        scenario_config = config.copy()
        scenario_config.update(scenario_params)
        
        scenarios[scenario_name] = generate_data_from_config(scenario_config)
    
    return scenarios

if __name__ == "__main__":
    # Example usage with different scenarios
    
    # Scenario 1: Strong confounding, strong prognostic effects, weak treatment effects
    print("Generating data with strong confounding and prognostic effects...")
    data = generate_data(
        N=10000,
        confounding_level=5.0,      # Strong confounding
        outcome_correlation=0.1
    )
    analyze_feature_treatment_relationships(data['X'], data['T'])
    experiment_correlation_vs_cosine(M=2)
    
    # Load config and generate data
    with open('MTML/config/data/synthetic_2d.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Generate single dataset
    data = generate_data_from_config(config)
    
    # Or generate all scenarios
    scenarios = generate_scenarios_from_config(config)