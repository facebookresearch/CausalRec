# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict
import logging
import pdb
import torch
import torch.nn as nn
import lightgbm as lgb
from torch.utils.data import DataLoader, Dataset
import numpy as np


class BaseModel(nn.Module):
    def __init__(self, config,
                 model_id="BaseModel",
                 task="classification",
                 device="cpu"):
        super().__init__()
        self.config = config
        self.model_id = model_id
        self.task = task
        self.device = device
        self.output_activation = self.get_output_activation(task)
        
    def get_output_activation(self, task):
        if task == "classification":
            return nn.Sigmoid()
        elif task == "regression":
            return nn.Identity()
    
    def count_parameters(self, count_embedding=True):
        total_params = 0
        for name, param in self.named_parameters(): 
            if not count_embedding and "embedding" in name:
                continue
            if param.requires_grad:
                total_params += param.numel()
        logging.info("Total number of parameters: {}.".format(total_params))
        
    def to_device(self, device):
        self.device = device
        self.to(device)
        

class TARNet(BaseModel):
    def __init__(self, config,
                 model_id="TARNet",
                 task="classification",
                 device="cpu"):
        super().__init__(config, model_id, task, device)
        self.is_trained = False
        layers = []
        prev_dim = config["input_dim"]
        hidden_dims = config["hidden_dims"]
        output_dim = config["output_dim"]
        dropout_rate = config.get("dropout_rate", 0.0)  # Default to 0 if not specified
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
        self.to_device(device)
        
    def forward(self, data):
        x = data["feature"]  # [batch, input_dim]
        output = self.network(x)
        pred = self.output_activation(output)
        return pred

    def predict(self, data, batch_size=16384):
        """Make predictions on the entire dataset using batches.
        
        Args:
            data: Dictionary containing input features and other data
            
        Returns:
            torch.Tensor: Model predictions for the entire dataset
        """
        self.eval()  # Set model to evaluation mode
        
        # Create DataLoader for batch processing
        dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
        
        predictions = []
        with torch.no_grad():  # Disable gradient computation
            for batch in dataloader:
                pred = self.forward(batch)
                predictions.append(pred)
                
        # Concatenate all batch predictions
        predictions = torch.cat(predictions, dim=0)
        
        return predictions
    
    
class GBDT:
    """LightGBM-based GBDT model"""
    def __init__(self, config, model_id="GBDT", task="classification"):
        
        self.model_id = model_id
        self.task = task
        
        # Default LightGBM parameters
        self.params = {
            'objective': 'binary' if task == 'classification' else 'regression',
            'metric': 'binary_logloss' if task == 'classification' else 'rmse',
            'boosting_type': 'gbdt',
            'max_depth': config.get('max_depth', 6),
            'max_leaves': config.get('max_leaves', 31),
            'learning_rate': config.get('gbdt_learning_rate', 0.05),
            'early_stopping_round': config.get('early_stopping_round', 20),
            'feature_fraction': config.get('feature_fraction', 0.9),
            'bagging_fraction': config.get('bagging_fraction', 0.8),
            'bagging_freq': config.get('bagging_freq', 5),
            'verbose': -1
        }
        
        # Update with any additional parameters from config
        if 'lgb_params' in config:
            self.params.update(config['lgb_params'])
            
        self.model = None
        
    def fit(self, X, y, valid_sets=None):
        """Train the model"""
        train_data = lgb.Dataset(X, label=y)
        self.model = lgb.train(
            self.params,
            train_data,
            valid_sets=valid_sets,
        )
        
    def predict(self, X):
        """Make predictions"""
        if self.model is None:
            raise RuntimeError("Model needs to be trained before prediction")
        return self.model.predict(X)

    
class OutcomePredictor():
    def __init__(self, model):
        self.model = model
    
    def predict_ctrl(self, data):
        pass
    
    def predict_trt(self, data):
        pass


class TARNetPredictor(OutcomePredictor):
    def __init__(self, model):
        super().__init__(model)
        
    def predict_ctrl(self, data):
        return self.model.predict(data)[:, 0]
    
    def predict_trt(self, data):
        return self.model.predict(data)[:, 1]
        
        
class DRlearner():
    def __init__(self, config):
        self.outcome_model = TARNet(config)
        self.outcome_predictor = TARNetPredictor(self.outcome_model)
        self.dr_model = GBDT(config)
    
    def pseudo_outcome(self, dataset, y, t):
        if y.ndim > 1:
            if y.shape[1] == 1:
                y = y.ravel()
            else:
                raise ValueError("y must be a 1D array")
        y_hat_ctrl = self.outcome_predictor.predict_ctrl(dataset).detach().cpu().numpy()
        y_hat_trt = self.outcome_predictor.predict_trt(dataset).detach().cpu().numpy()
        y_pred = t * y_hat_trt + (1 - t) * y_hat_ctrl
        p_hat = np.count_nonzero(t == 1) / len(t) # assume RCT data
        phi = (y - y_pred) * (t - p_hat) / (p_hat * (1 - p_hat)) + y_hat_trt - y_hat_ctrl
        return phi
    
    def fit(self, train_X, train_y, eval_X, eval_y):
        ### debug ###
        print(self.dr_model.params)
        train_data = lgb.Dataset(train_X, label=train_y)
        eval_data = lgb.Dataset(eval_X, label=eval_y)
        self.dr_model = lgb.train(
                self.dr_model.params,
                train_data,
                valid_sets=[train_data, eval_data],
            )
        
    def predict(self, X):
        return self.dr_model.predict(X)
        
        
        
        
        
