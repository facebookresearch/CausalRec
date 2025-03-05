import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict
from model.layer import (
    MultiTaskTreatmentPrediction,
    TaskEnhancedGate,
    TreatmentAwareUnit,
    TreatmentEnhancedGate
)

class ECUP(nn.Module):
    """
    Multiple Treatment Multiple-Outcome ECUP (Enhanced Chain Uplift Prediction) model.
    Uses a chain of treatment-enhanced and task-enhanced networks for predicting multiple outcomes.
    """

    def __init__(
        self,
        input_dim: int,
        num_treatments: int = 2,
        num_outcomes: int = 1,
        hidden_dims: List[int] = [64, 32],
        treatment_dim: int = 32,
        task_dim: int = 32,
        num_heads: int = 2,
        hidden_activations: str = "ReLU",
        net_dropout: float = 0,
        batch_norm: bool = False,
    ):
        super().__init__()
        self.num_treatments = num_treatments
        self.num_outcomes = num_outcomes

        # Feature and treatment embeddings
        self.feature_embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[-1])
        )
        self.treatment_proj = nn.Linear(num_treatments, treatment_dim)

        # Treatment-Enhanced Network (TENet)
        self.tau1 = TreatmentAwareUnit(
            feature_dim=hidden_dims[-1],  # Using the projected dimension
            treatment_dim=treatment_dim,
            tie_hidden_units=hidden_dims,
            hidden_activations=hidden_activations,
            dropout_rates=net_dropout,
            batch_norm=batch_norm,
        )
        self.tau2 = TreatmentAwareUnit(
            feature_dim=hidden_dims[-1],  # Using the projected dimension
            treatment_dim=treatment_dim,
            tie_hidden_units=hidden_dims,
            hidden_activations=hidden_activations,
            dropout_rates=net_dropout,
            batch_norm=batch_norm,
        )
        self.te_gate = TreatmentEnhancedGate(
            feature_dim=hidden_dims[-1],  # Using the projected dimension
            hidden_units=hidden_dims,
            hidden_activations=hidden_activations,
            dropout_rates=net_dropout,
            batch_norm=batch_norm,
        )

        # Task representation embedding
        self.task_embedding = nn.Parameter(torch.randn(num_outcomes, task_dim))

        # Task-Enhanced Network (TAENet)
        self.ta_gate = TaskEnhancedGate(
            input_dim=hidden_dims[-1],  # Using the projected dimension
            hidden_units=hidden_dims,
            hidden_activations=hidden_activations,
            dropout_rates=net_dropout,
            batch_norm=batch_norm,
            treatment_enhanced_dim=hidden_dims[-1],  # Using the projected dimension
            num_heads=num_heads,
        )

        # Final prediction network
        self.task_prediction = MultiTaskTreatmentPrediction(
            input_dim=hidden_dims[-1],  # Using the projected dimension
            num_treatments=num_treatments,
            num_outcomes=num_outcomes,
            hidden_units=hidden_dims,
            hidden_activations=hidden_activations,
            dropout_rates=net_dropout,
            batch_norm=batch_norm,
        )

    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass of the ECUP model.
        
        Args:
            features: Dictionary containing input features
            
        Returns:
            Tensor of shape [batch_size, num_outcomes, num_treatments] containing predictions
        """
        x = features["feature"]  # Using "feature" key instead of "dense_features"
        batch_size = x.size(0)

        # Project input features
        x_embedding = self.feature_embedding(x)  # [batch_size, input_dim]

        # Process all treatments at once
        treatment_indices = torch.arange(self.num_treatments, device=x.device)
        treatment_onehot = F.one_hot(treatment_indices, num_classes=self.num_treatments).float()
        treatment_emb = self.treatment_proj(treatment_onehot)  # [num_treatments, treatment_dim]

        # Expand for batch processing
        treatment_emb = treatment_emb.unsqueeze(0).expand(batch_size, -1, -1)
        x_embedding_expanded = x_embedding.unsqueeze(1).expand(-1, self.num_treatments, -1)

        # Treatment-aware processing
        tau1 = self.tau1(x_embedding_expanded, treatment_emb)
        tau2 = self.tau2(x_embedding_expanded, treatment_emb)
        enhanced_features = self.te_gate(tau1, tau2)

        # Task-aware processing
        task_emb = self.task_embedding.unsqueeze(0).expand(batch_size, -1, -1)
        task_emb_expanded = task_emb.unsqueeze(2).expand(-1, -1, self.num_treatments, -1)
        enhanced_features_expanded = enhanced_features.unsqueeze(1).expand(
            -1, self.num_outcomes, -1, -1
        )

        # Apply task-enhanced gating
        scales = self.ta_gate(task_emb_expanded, enhanced_features_expanded)
        scaled_enhanced_features = enhanced_features_expanded * scales

        # Get final predictions
        predictions = self.task_prediction(scaled_enhanced_features)

        return predictions  # [batch_size, num_outcomes, num_treatments]

