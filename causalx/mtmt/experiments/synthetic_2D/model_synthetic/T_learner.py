import torch
import torch.nn as nn

class SingleTlearner(nn.Module):
    """Single T-learner for one outcome"""
    def __init__(self, input_dim, num_treatment, hidden_dims=[64, 32]):
        super(SingleTlearner, self).__init__()
        
        self.num_treatment = num_treatment
        # Create separate networks for each treatment
        self.networks = nn.ModuleList([
            self._build_network(input_dim, hidden_dims) 
            for _ in range(num_treatment)
        ])
    
    def _build_network(self, input_dim, hidden_dims):
        layers = []
        prev_dim = input_dim
        
        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        
        return nn.Sequential(*layers)
    
    def forward(self, x, t):
        """
        Args:
            x: Feature tensor [batch_size, input_dim]
            t: Treatment tensor [batch_size]
        Returns:
            Predicted outcomes [batch_size, 1]
        """
        batch_size = x.shape[0]
        predictions = torch.zeros(batch_size, 1).to(x.device)
        
        # For each treatment value, use corresponding network to predict
        for treatment in range(self.num_treatment):
            mask = (t == treatment)
            if mask.any():
                predictions[mask] = self.networks[treatment](x[mask])
        
        return predictions

class Tlearner(nn.Module):
    """T-learner for multiple outcomes and treatments"""
    def __init__(self, input_dim, num_outcome, num_treatment, task="regression"):
        super(Tlearner, self).__init__()
        
        self.input_dim = input_dim
        self.num_outcome = num_outcome
        self.num_treatment = num_treatment
        self.task = task
        
        # Create separate T-learner for each outcome
        self.outcome_learners = nn.ModuleList([
            SingleTlearner(input_dim, num_treatment)
            for _ in range(num_outcome)
        ])
    
    def forward(self, data_dict):
        """
        Args:
            data_dict: Dictionary containing:
                - feature: [batch_size, input_dim]
                - treatment: [batch_size]
        Returns:
            Predicted outcomes [batch_size, num_outcome]
        """
        x = data_dict["feature"]
        t = data_dict["treatment"]
        batch_size = x.shape[0]
        
        # Get predictions for each outcome
        predictions = torch.zeros(batch_size, self.num_outcome).to(x.device)
        for outcome_idx in range(self.num_outcome):
            predictions[:, outcome_idx] = self.outcome_learners[outcome_idx](x, t).squeeze()
        
        return predictions
    
    def predict_counterfactuals(self, x):
        """
        Predict outcomes for all possible treatments
        Args:
            x: Feature tensor [batch_size, input_dim]
        Returns:
            Counterfactual predictions [batch_size, num_treatment, num_outcome]
        """
        batch_size = x.shape[0]
        device = x.device
        counterfactuals = torch.zeros(batch_size, self.num_treatment, self.num_outcome).to(device)
        
        # For each treatment value
        for t in range(self.num_treatment):
            # Create treatment tensor
            t_tensor = torch.full((batch_size,), t, dtype=torch.long).to(device)
            # Get predictions for all outcomes
            data_dict = {"feature": x, "treatment": t_tensor}
            predictions = self.forward(data_dict)
            counterfactuals[:, t, :] = predictions
        
        return counterfactuals 