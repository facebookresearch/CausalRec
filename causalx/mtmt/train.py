import logging
from datetime import datetime
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from typing import Dict
from dataset.dataset import UpliftDataset
from model.model import MLP, PLE2D, S_MMoE, Slearner_uplift, Vanilla_MMoE, TARNet, Slearner, MultiTaskMoE, MultiTaskMoEDANN, MultiTaskMoETransformer, Tlearner
from torcheval.metrics import BinaryNormalizedEntropy, MeanSquaredError
from trainer.trainer import TrainerUnit, Trainer
from torchtnt.utils.loggers.tensorboard import TensorBoardLogger
from torchtnt.utils import env
from torchtnt.utils import init_from_env
import hydra
from omegaconf import DictConfig, OmegaConf
import pdb
import json
from util.utility import eval_multi_trt, calculate_stats, eval_multi_trt_ctcvr
import numpy as np
import torch
import random

logger = logging.getLogger()

def set_seed(seed: int):
    env.seed(seed)
    # Ensure deterministic behavior in PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# single run command:
# python train.py

# sweep commands:
# python train.py -m hydra/sweeper/sampler=$sampler_choice objective=$metric
# sampler_choice = [random, tpe, grid]
# metric = [auuc_outcome_0, auuc_outcome_1, auuc_outcome_no_trt1_0, auuc_outcome_no_trt1_1]

# use tpe
@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> float:
    results_lst = []
    for seed in cfg.seeds:
        results = train(cfg)
        results_lst.append(results)
    mean_result, std_result = calculate_stats(results_lst)
    final_result = {
        "mean_result": mean_result,
        "std_result": std_result,
        "raw_results_lst": results_lst
    }
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    results_file = os.path.join(output_dir, "results")
    with open(results_file, "w") as file:
        json.dump(final_result, file, indent=4)
    
    logger.info(f"Saving results into {results_file}")
    logger.info(f"Objective: {cfg.objective}")
  
    return final_result["mean_result"][cfg.objective]
        
        
def train(cfg : DictConfig) -> Dict:
    # Set seed for reproducibility
    set_seed(cfg.seeds[0])  # Use the first seed from the config
    
    # prepare dataset
    data_name = cfg.data.name
    if data_name == "criteo":
        data = pd.read_csv(cfg.data.path)
        train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)
    elif data_name == "mtlift":
        train_df = pd.read_csv(cfg.data.train_path)
        test_df = pd.read_csv(cfg.data.test_path)
        if "ctcvr" in cfg.data.outcome_cols:
            train_df["ctcvr"] = train_df["click"] * train_df["conversion"]
            test_df["ctcvr"] = test_df["click"] * test_df["conversion"]
    else:
        raise ValueError(f"Unsupported dataset {data_name}")
    
    # Feature normalization, outcome standardization
    data_config = cfg.data
    scaler_type = cfg.data.scaler  # Get the scaler type

    # Initialize the appropriate scaler
    if scaler_type == "minmax":
        scaler = MinMaxScaler(feature_range=(0, 1))
    elif scaler_type == "standard":
        scaler = StandardScaler()
    else:
        raise ValueError("Invalid scaler type specified in the config. Use 'minmax' or 'standard'.")

    # Setup device
    device = init_from_env()
    logger.info(f"Running on device: {device}")
    
    # Add treatment column for certain models
    if cfg.model.name.startswith("S"):
        data_config["feature_cols"].append(data_config["treatment_col"])
    
    # Prepare datasets
    train_dataset = UpliftDataset(train_df, data_config, device)
    test_dataset = UpliftDataset(test_df, data_config, device)
    train_dataset.feature_fit(scaler)
    train_dataset.feature_transform(scaler)
    test_dataset.feature_transform(scaler)
    
    # Construct model
    num_features = len(data_config["feature_cols"])
    num_outcome = len(data_config["outcome_cols"])
    num_treatment = data_config["num_treatment"]
    model_config = cfg.model
    model_name = model_config["name"]
    
    # Model selection
    if model_name == "test":
        model = MLP(input_dim=num_features, output_dim=num_treatment)
    elif model_name == "TARNet":
        model = TARNet(input_dim=num_features, num_outcome=num_outcome, num_treatment=num_treatment)
    elif model_name == "Slearner":
        model = Slearner_uplift(input_dim=num_features, num_outcome=num_outcome)
    elif model_name == "Tlearner":
        model = Tlearner(input_dim=num_features, num_outcome=num_outcome, num_treatment=num_treatment)
    elif model_name == "S_MMoE":
        model = S_MMoE(
                 num_outcome=num_outcome,
                 num_experts=model_config["num_experts"],
                 expert_hidden_units=model_config["expert_hidden_units"],
                 gate_hidden_units=model_config["gate_hidden_units"],
                 tower_hidden_units=model_config["tower_hidden_units"],
                 hidden_activations=model_config["hidden_activations"],
                 net_dropout=model_config["net_dropout"],
                 batch_norm=model_config["batch_norm"],
                 input_dim=num_features,
                 task=model_config["task"]
                )
    elif model_name == "Vanilla_MMoE":
        model = Vanilla_MMoE(
                 num_outcomes=num_outcome,
                 num_treatments=num_treatment,
                 num_experts=model_config["num_experts"],
                 expert_hidden_units=model_config["expert_hidden_units"],
                 gate_hidden_units=model_config["gate_hidden_units"],
                 tower_hidden_units=model_config["tower_hidden_units"],
                 hidden_activations=model_config["hidden_activations"],
                 net_dropout=model_config["net_dropout"],
                 batch_norm=model_config["batch_norm"],
                 input_dim=num_features,
                 task=model_config["task"]
                )
    elif model_name == "PLE2D":
        model = PLE2D(
                 num_layers=model_config["num_layers"],
                 expert_type=model_config["expert_type"],
                 dcnv2_model_structure=model_config["dcnv2_model_structure"],
                 dcnv2_num_cross_layers=model_config["dcnv2_num_cross_layers"],
                 num_shared_experts=model_config["num_shared_experts"],
                 num_outcome_shared_experts=model_config["num_outcome_shared_experts"],
                 num_treatment_shared_experts=model_config["num_treatment_shared_experts"],
                 num_specific_experts=model_config["num_specific_experts"],
                 num_outcomes=num_outcome,
                 num_treatments=num_treatment,
                 expert_hidden_units=model_config["expert_hidden_units"],
                 gate_hidden_units=model_config["gate_hidden_units"],
                 tower_hidden_units=model_config["tower_hidden_units"],
                 hidden_activations=model_config["hidden_activations"],
                 net_dropout=model_config["net_dropout"],
                 batch_norm=model_config["batch_norm"],
                 input_dim=num_features,
                 task=model_config["task"]
                )
    elif model_name == "MMoE":
        model = MultiTaskMoE(
                num_outcomes=num_outcome,
                num_treatments=num_treatment,
                num_experts=model_config["num_experts"],
                expert_hidden_units=model_config["expert_hidden_units"],
                gate_hidden_units=model_config["gate_hidden_units"],
                hidden_activations=model_config["hidden_activations"],
                net_dropout=model_config["net_dropout"],
                batch_norm=model_config["batch_norm"],
                input_dim=num_features,
                task=model_config["task"]
                )
    elif model_name == "MMoE_DANN":
        model = MultiTaskMoEDANN(
                num_outcomes=num_outcome,
                num_treatments=num_treatment,
                num_experts=model_config["num_experts"],
                expert_hidden_units=model_config["expert_hidden_units"],
                gate_hidden_units=model_config["gate_hidden_units"],
                hidden_activations=model_config["hidden_activations"],
                net_dropout=model_config["net_dropout"],
                batch_norm=model_config["batch_norm"],
                input_dim=num_features,
                task=model_config["task"]
                )
    elif model_name == "MMoE_Transformer":
        model = MultiTaskMoETransformer(
                num_outcomes=num_outcome,
                num_treatments=num_treatment,
                num_experts=model_config["num_experts"],
                expert_hidden_units=model_config["expert_hidden_units"],
                gate_hidden_units=model_config["gate_hidden_units"],
                hidden_activations=model_config["hidden_activations"],
                net_dropout=model_config["net_dropout"],
                batch_norm=model_config["batch_norm"],
                input_dim=num_features,
                task=model_config["task"],
                gating_output_activation=model_config.get("gating_output_activation", "softmax"),
                top_k=model_config.get("top_k", None)
                )
    else:
        raise ValueError("Model type not supported")
    
    # Tensorboard logging
    tb_logger = TensorBoardLogger(
                    path=os.path.join(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir, "tensorboard_logs"),
                )
    
    # Loss and training configurations
    loss_config = cfg.model.loss_config
    train_config = cfg.train

    # Standardized checkpoint configuration
    checkpoint_config = {
        "save_dir": os.path.join(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir, "checkpoints"),
        "save_interval": train_config.get("checkpoint_interval", 5),
        "max_keep": train_config.get("max_checkpoints", 3)
    }

    # Ensure checkpoint directory exists
    os.makedirs(checkpoint_config["save_dir"], exist_ok=True)

    # Create trainer unit
    unit = TrainerUnit(
        module=model,
        tb_logger=tb_logger,
        log_every_n_steps=train_config["log_every_n_steps"],
        lr=train_config["lr"],
        optimizer_name=train_config["optimizer_name"],
        weight_decay=train_config["weight_decay"],
        loss_config=loss_config,
        device=device,
        checkpoint_config=checkpoint_config
    )

    # Create trainer
    trainer = Trainer(
        unit=unit,
        train_config=train_config,
        device=device
    )

    # Resume from checkpoint if specified
    resume_checkpoint = cfg.train.get("resume_from_checkpoint", None)
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

    plot_config = {
        "save_dir": os.path.join(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir, "plots"),
    }
    
    logger.info(f"checkpoint paths: {checkpoint_paths}")
    logger.info(f"losses: {losses}")

    # Ensure checkpoint directory exists
    os.makedirs(plot_config["save_dir"], exist_ok=True)


    # Evaluate model by AUUCs
    trt_list = np.arange(num_treatment)
    metric_type = "average"
    results = {
        "train_loss": losses[0],
        "eval_loss": losses[1]
    }
    model_type = (
                "s" if model_name.startswith("S") else 
                "adversarial" if model_name == "MMoE_DANN" else 
                "transformer" if model_name == "MMoE_Transformer" else
                "t"
                )
    if "ctcvr" in cfg.data.outcome_cols:
        data_config.outcome_cols = ["ctcvr", "conversion"] # handle ctcvr and conversion auuc eval separately
        test_dataset = UpliftDataset(test_df, data_config, device)
        test_dataset.feature_transform(scaler)
        click_test_df = test_df[test_df["click"]==1] # to eval cvr in the click space
        click_test_dataset = UpliftDataset(click_test_df, data_config, device)
        click_test_dataset.feature_transform(scaler)
        ctcvr_auuc, all_ctcvr_auuc, average_auuc_exclude_1 = eval_multi_trt_ctcvr(model, test_dataset, 0, trt_list, metric_type, model_type, plot=cfg.plot, save_path=plot_config["save_dir"])
        conv_auuc, all_conv_auuc, average_auuc_exclude_1 = eval_multi_trt(model, click_test_dataset, 1, trt_list, metric_type, model_type, plot=cfg.plot, save_path=plot_config["save_dir"])
    
        results["auuc_outcome_0"] = ctcvr_auuc
        results["auuc_outcome_1"] = conv_auuc
        results["all_auuc_outcome_0"] = all_ctcvr_auuc
        results["all_auuc_outcome_1"] = all_conv_auuc
        
        if cfg.eval_train:
            click_train_df = train_df[train_df["click"]==1] # to eval cvr in the click space
            click_train_dataset = UpliftDataset(click_train_df, data_config, device)
            click_train_dataset.feature_transform(scaler)
            ctcvr_auuc_train, all_ctcvr_auuc_train, average_auuc_exclude_1_train = eval_multi_trt_ctcvr(model, train_dataset, 0, trt_list, metric_type, model_type, plot=cfg.plot, save_path=plot_config["save_dir"])
            conv_auuc_train, all_conv_auuc_train, average_auuc_exclude_1_train = eval_multi_trt(model, click_train_dataset, 1, trt_list, metric_type, model_type, plot=cfg.plot, save_path=plot_config["save_dir"])
            
            results["auuc_outcome_0_train"] = ctcvr_auuc_train
            results["auuc_outcome_1_train"] = conv_auuc_train
            results["all_auuc_outcome_0_train"] = all_ctcvr_auuc_train
            results["all_auuc_outcome_1_train"] = all_conv_auuc_train
    else:
        for outcome_index in np.arange(num_outcome):
            # Compute AUUCs
            auuc, all_auuc, average_auuc_exclude_1 = eval_multi_trt(model, test_dataset, outcome_index, trt_list, metric_type, model_type, plot=cfg.plot, save_path=plot_config["save_dir"])
            
            avg_outcome_key = "auuc_outcome_" + str(outcome_index)
            all_outcome_key = "all_auuc_outcome_" + str(outcome_index)
            all_outcome_but1_key = "auuc_outcome_no_trt1_" + str(outcome_index)
            results[avg_outcome_key] = auuc
            results[all_outcome_key] = all_auuc
            results[all_outcome_but1_key] = average_auuc_exclude_1
    return results


if __name__ == "__main__":
    main()