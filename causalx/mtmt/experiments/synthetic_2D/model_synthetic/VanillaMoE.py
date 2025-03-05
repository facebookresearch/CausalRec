import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.insert(0, '../../')
from model.layer import MMoE_Layer, MLP_Block, get_output_activation

class VanillaMoE(nn.Module):
    """Mixture of Experts for binary treatment with multiple outcomes"""
    def __init__(self,
                 input_dim,
                 num_outcome,
                 num_experts=4,
                 expert_hidden_units=[512, 256, 128],
                 gate_hidden_units=[128, 64],
                 tower_hidden_units=[128, 64],
                 hidden_activations="ReLU",
                 net_dropout=0,
                 batch_norm=False,
                 task="regression"):
        super().__init__()
        self.num_outcomes = num_outcome
        self.output_activation = get_output_activation(task)
        
        # Single MMoE layer that handles all outcomes
        self.mmoe = MMoE_Layer(
            num_experts=num_experts,
            num_tasks=num_outcome,  # One task per outcome
            input_dim=input_dim,
            expert_hidden_units=expert_hidden_units,
            gate_hidden_units=gate_hidden_units,
            hidden_activations=hidden_activations,
            net_dropout=net_dropout,
            batch_norm=batch_norm
        )
        
        # Create towers for each outcome
        self.towers = nn.ModuleList([
            MLP_Block(
                input_dim=expert_hidden_units[-1],
                output_dim=1,
                hidden_units=tower_hidden_units,
                hidden_activations=hidden_activations,
                dropout_rates=net_dropout,
                batch_norm=batch_norm
            )
            for _ in range(num_outcome)
        ])

    def forward(self, data_dict):
        """
        Args:
            data_dict: Dictionary containing:
                - feature: [batch_size, input_dim]
                - treatment: [batch_size] (should be binary 0/1)
        Returns:
            Predicted outcomes [batch_size, num_outcome]
        """
        x = data_dict["feature"]
        batch_size = x.shape[0]
        device = x.device
        
        # Get MMoE outputs
        mmoe_outputs = self.mmoe(x)  # List of [batch, expert_dim]
        
        # Pass through outcome-specific towers
        outcome_preds = []
        for outcome_idx in range(self.num_outcomes):
            tower_out = self.towers[outcome_idx](mmoe_outputs[outcome_idx])
            if self.output_activation is not None:
                tower_out = self.output_activation(tower_out)
            outcome_preds.append(tower_out)
        
        # Combine predictions for all outcomes
        predictions = torch.cat(outcome_preds, dim=1)  # [batch, num_outcome]
        
        return predictions

    def predict_counterfactuals(self, x):
        """
        Not implemented for this model since it should be trained separately 
        for each treatment group
        """
        raise NotImplementedError(
            "This model should be trained separately for each treatment group. "
            "Use two separate models for counterfactual predictions."
        ) 