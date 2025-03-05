# /Users/yilingliu/Downloads/MTML/utility.py

import logging
import torch
from torch import nn
from torchtnt.utils import init_from_env
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from causalml.metrics import auuc_score
from causalml.metrics.visualize import plot_gain
import os
import sys
from torch.utils.data import DataLoader
from geomloss import SamplesLoss


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
    # Generate base features
    if Sigma_X is None:
        X_base = np.random.randn(n_samples, n_features)
    else:
        X_base = np.random.multivariate_normal(
            mean=np.zeros(n_features),
            cov=Sigma_X,
            size=n_samples
        )
    
    # Create and initialize a simple MLP for feature transformation
    class FeatureTransformer(nn.Module):
        def __init__(self, input_dim, hidden_dim=64):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, input_dim)
            
        def forward(self, x):
            h = torch.tanh(self.fc1(x))  # Using tanh for bounded nonlinearity
            return self.fc2(h)
    
    # Initialize and apply the transformer
    transformer = FeatureTransformer(n_features)
    X_torch = torch.FloatTensor(X_base)
    with torch.no_grad():
        X = transformer(X_torch).numpy()

    return X, X_base

class TreatmentMLP(nn.Module):
    def __init__(self, input_dim, num_treatments, hidden_dim=32):
        super().__init__()
        self.fc1 = nn.Linear(input_dim + num_treatments, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        
    def forward(self, x, t_onehot):
        inputs = torch.cat([x, t_onehot], dim=1)
        h = F.relu(self.fc1(inputs))
        out = self.fc2(h)
        return out.squeeze(-1)