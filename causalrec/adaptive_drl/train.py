# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

# pyre-strict
import logging
from datetime import datetime
import os
import pandas as pd
from causalml.metrics import auuc_score
from causalml.metrics.visualize import plot_gain
from dataset.dataset import build_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from typing import Dict, Tuple
from dataset.dataset import UpliftDataset
from torch.utils.data import Dataset
from model.model import DRlearner, TARNet
from torcheval.metrics import BinaryNormalizedEntropy, MeanSquaredError
from trainer.trainer import TrainerUnit, Trainer
from torchtnt.utils.loggers.tensorboard import TensorBoardLogger
from torchtnt.utils import env
from torchtnt.utils import init_from_env
import hydra
from omegaconf import DictConfig, OmegaConf
import pdb
import json
import numpy as np
import torch
import random

logger = logging.getLogger(__name__)

def set_seed(seed: int):
    env.seed(seed)
    # Ensure deterministic behavior in PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup(cfg : DictConfig) -> Tuple[torch.device, Dataset, Dataset, TARNet]:
    # Set seed for reproducibility
    set_seed(cfg.seeds[0])  # Use the first seed from the config
    
    # Setup device
    device = init_from_env()
    logger.info(f"Running on device: {device}")
    
    # Prepare dataset
    data_name = cfg.data.name
    logger.info(f"Loading dataset: {data_name}")
    data = pd.read_csv(cfg.data.path)
    train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)
    data_cfg = OmegaConf.to_container(cfg.data)
    train_dataset, processor = build_dataset(train_df, data_cfg, True, device, processors=None)
    test_dataset, _ = build_dataset(test_df, data_cfg, False, device, processor)
    
    return device, train_dataset, test_dataset

def train_tarnet(cfg : DictConfig, tarnet: TARNet, device: torch.device, train_dataset: Dataset, test_dataset: Dataset) -> TARNet:
    # Tensorboard logging
    tb_logger = TensorBoardLogger(
                    path=os.path.join(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir, "tensorboard_logs"),
                )
    
    # Loss and training configurations
    loss_config = cfg.model.loss_config
    trainer_config = cfg.trainer

    # Standardized checkpoint configuration
    checkpoint_config = {
        "save_dir": os.path.join(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir, "checkpoints"),
        "save_interval": trainer_config.get("checkpoint_interval", 5),
        "max_keep": trainer_config.get("max_checkpoints", 3)
    }

    # Ensure checkpoint directory exists
    os.makedirs(checkpoint_config["save_dir"], exist_ok=True)

    # Create trainer unit
    unit = TrainerUnit(
        module=tarnet,
        tb_logger=tb_logger,
        log_every_n_steps=trainer_config["log_every_n_steps"],
        lr=trainer_config["lr"],
        optimizer_name=trainer_config["optimizer_name"],
        weight_decay=trainer_config["weight_decay"],
        loss_config=loss_config,
        device=device,
        checkpoint_config=checkpoint_config
    )

    # Create trainer
    trainer = Trainer(
        unit=unit,
        trainer_config=trainer_config,
        device=device
    )

    # Resume from checkpoint if specified
    resume_checkpoint = trainer_config.get("resume_from_checkpoint", None)
    if resume_checkpoint:
        logger.info(f"Resuming training from checkpoint: {resume_checkpoint}")
        try:
            unit.load_checkpoint(resume_checkpoint)
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise

    # Training
    losses = [-1, -1]
    checkpoint_paths = []
    if not cfg.debug:
        checkpoint_paths, losses = trainer.train(train_dataset, test_dataset)
        trainer.unit.module.is_trained = True

    plot_config = {
        "save_dir": os.path.join(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir, "plots"),
    }
    
    logger.info(f"checkpoint paths: {checkpoint_paths}")
    logger.info(f"losses: {losses}")

    # Ensure checkpoint directory exists
    os.makedirs(plot_config["save_dir"], exist_ok=True)

    return tarnet

def train_drlearner(cfg : DictConfig, device: torch.device, train_dataset: Dataset, test_dataset: Dataset) -> DRlearner:
    dr_learner = DRlearner(cfg.model)
    train_tarnet(cfg, dr_learner.outcome_model, device, train_dataset, test_dataset)
    # # fake training
    # dr_learner.outcome_model.is_trained = True
    if dr_learner.outcome_model.is_trained:
        train_X = train_dataset.feature
        eval_X = test_dataset.feature
        train_y = dr_learner.pseudo_outcome(train_dataset, train_dataset.outcome, train_dataset.trt)
        eval_y = dr_learner.pseudo_outcome(test_dataset, test_dataset.outcome, test_dataset.trt)
        dr_learner.fit(train_X, train_y, eval_X, eval_y)
    else:
        raise ValueError("Outcome model is not trained")
    
    return dr_learner

def eval_AUUC(uplift_scores: np.ndarray, treatment: np.ndarray, outcome: np.ndarray) -> float:
    """
    Evaluate Area Under the Uplift Curve (AUUC) on test dataset. AUUC measures a model's ability 
    to identify individuals who responds positively to treatment, and higher AUUC indicates better 
    targeting of treatments to users.
    
    Args:
        uplift_scores: Array of predicted treatment effects
        treatment: Array of treatment assignments (0 or 1)
        outcome: Array of observed outcomes
        
    Returns:
        float: Normalized AUUC score between 0 and 1
        
    Raises:
        ValueError: If input arrays have different lengths or invalid values
    """
    # Input validation
    if not (len(uplift_scores) == len(treatment) == len(outcome)):
        raise ValueError("Input arrays must have the same length")
        
    if not np.all(np.isin(treatment, [0,1])):
        raise ValueError("Treatment must be binary (0 or 1)")
    
    if treatment.ndim > 1:
        if treatment.shape[1] == 1:
            treatment = treatment.ravel()
        else:
            raise ValueError("Treatment must be a 1D array")
    if outcome.ndim > 1:
        if outcome.shape[1] == 1:
            outcome = outcome.ravel()
        else:
            raise ValueError("Outcome must be a 1D array")
    if uplift_scores.ndim > 1:
        if uplift_scores.shape[1] == 1:
            uplift_scores = uplift_scores.ravel()
        else:
            raise ValueError("Uplift scores must be a 1D array")
        
    # Create dataframe for evaluation
    data = {
        'treatment': treatment,
        'outcome': outcome,
        'uplift_score': uplift_scores
    }
    df = pd.DataFrame(data)
    
    # Calculate normalized AUUC 
    auuc = auuc_score(
        df,
        outcome_col='outcome',
        treatment_col='treatment',
        normalize=True
    )
    
    return auuc["uplift_score"]

    
@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> float:
    set_seed(cfg.seeds[0])
    device, train_dataset, test_dataset = setup(cfg)
    cfg.model.input_dim = train_dataset.feature.shape[1] # assume all features are floats
    dr_learner = train_drlearner(cfg, device, train_dataset, test_dataset)
    uplift_scores = dr_learner.predict(test_dataset.feature)
    auuc = eval_AUUC(uplift_scores, test_dataset.trt, test_dataset.outcome)
    logger.info(f"AUUC: {auuc}")

if __name__ == "__main__":
    main()
