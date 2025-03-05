import torch
import torch.nn as nn
import torch.nn.functional as F
from geomloss import SamplesLoss

class SingleTARNet(nn.Module):
    """Single TARNet for one outcome"""
    def __init__(self, input_dim, num_treatment, hidden_dims=[64, 32]):
        super(SingleTARNet, self).__init__()
        self.num_treatment = num_treatment
        
        # Feature representation network (shared across treatments)
        layers = []
        current_dim = input_dim
        for hidden_dim in hidden_dims[:-1]:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.1)
            ])
            current_dim = hidden_dim
        
        self.feature_net = nn.Sequential(*layers)
        self.representation_dim = hidden_dims[-1]
        
        # Treatment-specific networks
        self.treatment_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(current_dim, self.representation_dim),
                nn.ReLU(),
                nn.BatchNorm1d(self.representation_dim),
                nn.Dropout(0.1),
                nn.Linear(self.representation_dim, 1)
            ) for _ in range(num_treatment)
        ])
    
    def forward(self, x, t):
        """
        Args:
            x: Feature tensor [batch_size, input_dim]
            t: Treatment tensor [batch_size]
        Returns:
            Predicted outcomes [batch_size, 1]
        """
        # Get shared representation
        shared_repr = self.feature_net(x)
        
        # Get predictions for each treatment
        batch_size = x.size(0)
        predictions = torch.zeros(batch_size, 1).to(x.device)
        
        for i in range(self.num_treatment):
            mask = (t == i)
            if mask.any():
                predictions[mask] = self.treatment_nets[i](shared_repr[mask])
        
        return predictions
    
    def get_representations(self, x):
        """Get shared representations for input features"""
        return self.feature_net(x)

class TARNet(nn.Module):
    """TARNet with MMD regularization for multiple treatments and outcomes"""
    def __init__(self, input_dim, num_outcome, num_treatment, hidden_dims=[64, 32], task="regression"):
        super(TARNet, self).__init__()
        self.num_treatment = num_treatment
        self.num_outcome = num_outcome
        self.task = task
        
        # Create separate TARNet for each outcome
        self.outcome_nets = nn.ModuleList([
            SingleTARNet(input_dim, num_treatment, hidden_dims)
            for _ in range(num_outcome)
        ])
        
        # Initialize MMD loss
        self.mmd_loss = SamplesLoss(loss="energy", backend="tensorized")

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
            predictions[:, outcome_idx] = self.outcome_nets[outcome_idx](x, t).squeeze()
        
        return predictions

    def compute_mmd_loss(self, shared_repr, treatments):
        """
        Compute MMD loss between treatment groups using energy distance
        Args:
            shared_repr: Shared representations [batch_size, repr_dim]
            treatments: Treatment assignments [batch_size]
        Returns:
            MMD loss
        """
        mmd_loss = 0
        n_pairs = 0
        
        # Compute MMD between all pairs of treatment groups
        for i in range(self.num_treatment):
            for j in range(i+1, self.num_treatment):
                mask_i = (treatments == i)
                mask_j = (treatments == j)
                
                if mask_i.any() and mask_j.any():
                    repr_i = shared_repr[mask_i]
                    repr_j = shared_repr[mask_j]
                    
                    # Compute energy distance between distributions
                    mmd_loss += self.mmd_loss(repr_i, repr_j)
                    n_pairs += 1
        
        return mmd_loss / max(n_pairs, 1)  # Normalize by number of pairs

    def get_representations(self, x):
        """Get shared representations for all outcomes
        Args:
            x: Feature tensor [batch_size, input_dim]
        Returns:
            List of representations, one for each outcome
        """
        representations = []
        for outcome_net in self.outcome_nets:
            representations.append(outcome_net.get_representations(x))
        return representations

    def compute_all_mmd_losses(self, x, treatments):
        """Compute MMD losses for all outcomes
        Args:
            x: Feature tensor [batch_size, input_dim]
            treatments: Treatment assignments [batch_size]
        Returns:
            List of MMD losses, one for each outcome
        """
        mmd_losses = []
        representations = self.get_representations(x)
        
        for outcome_idx in range(self.num_outcome):
            mmd_loss = self.compute_mmd_loss(representations[outcome_idx], treatments)
            mmd_losses.append(mmd_loss)
            
        return mmd_losses

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