import logging
from datetime import datetime
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from typing import Dict
from dataset.dataset import UpliftDataset
from model.model import MLP, PLE2D, PLE2D_DA, S_MMoE, Vanilla_MMoE, TARNet, Slearner, MultiTaskMoE, MultiTaskMoEDANN, MultiTaskMoETransformer
from trainer.trainer import TrainerUnit, Trainer
from torchtnt.utils.loggers.tensorboard import TensorBoardLogger
from torchtnt.utils import env
from torchtnt.utils import init_from_env
import hydra
from omegaconf import DictConfig, OmegaConf
import json
from util.utility import calculate_stats
import numpy as np
import torch
import random
from experiments.synthetic_2D.DGP import generate_data
import torch.nn.functional as F

logger = logging.getLogger()

def set_seed(seed: int):
    """
    Set random seed for reproducibility across all random number generators
    Args:
        seed: random seed
    """
    # Python's random module
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU
    
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
    set_seed(cfg.seed)
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
    # prepare dataset
    data_name = cfg.data.name
    if data_name == "synthetic2d":
        # Generate synthetic data
        synthetic_data = generate_data(
            N=cfg.data.num_samples,
            d=cfg.data.num_features,
            num_treatments=cfg.data.num_treatment,
            num_outcomes=len(cfg.data.outcome_cols),
            confounding_level=cfg.data.confounding_level,
            outcome_correlation=cfg.data.outcome_correlation,
            seed=cfg.seed
        )
        
        # Convert to dataframes
        X, T, Y = synthetic_data['X'], synthetic_data['T'], synthetic_data['Y']
        
        # Create train/test split
        X_train, X_test, T_train, T_test, Y_train, Y_test = train_test_split(
            X, T, Y, test_size=0.2, random_state=cfg.seed
        )
        
        # Create dataframes
        train_df = pd.DataFrame(X_train, columns=[f'x{i}' for i in range(X.shape[1])])
        train_df['treatment'] = T_train
        for i in range(Y.shape[1]):
            train_df[f'y{i}'] = Y_train[:, i]
            
        test_df = pd.DataFrame(X_test, columns=[f'x{i}' for i in range(X.shape[1])])
        test_df['treatment'] = T_test
        for i in range(Y.shape[1]):
            test_df[f'y{i}'] = Y_test[:, i]
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
    elif scaler_type == "none":  # Skip scaling for synthetic data
        scaler = None
    else:
        raise ValueError("Invalid scaler type specified in the config. Use 'minmax', 'standard', or 'none'.")

    # Setup device
    device = init_from_env()
    logger.info(f"Running on device: {device}")
    
    # Add treatment column for certain models
    if cfg.model.name.startswith("S"):
        data_config["feature_cols"].append(data_config["treatment_col"])
    
    # Prepare datasets
    train_dataset = UpliftDataset(train_df, data_config, device)
    test_dataset = UpliftDataset(test_df, data_config, device)

    # Only apply scaling if a scaler is defined
    if scaler is not None:
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
        model = Slearner(input_dim=num_features, num_outcome=num_outcome)
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
    elif model_name == "ple2d_DA":
        model = PLE2D_DA(
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

    # Evaluation
    model.eval()
    with torch.no_grad():
        # Calculate MSE for each treatment-outcome combination
        num_outcomes = len(data_config["outcome_cols"])
        num_treatments = data_config["num_treatment"]
        mse_matrix = torch.zeros(num_treatments, num_outcomes)
        
        # For test set evaluation using UpliftDataset format
        batch = {
            "feature": test_dataset.features,
            "treatment": test_dataset.treatments,
            "outcome": test_dataset.outcomes
        }
        all_preds = model(batch)  # [batch, num_outcomes, num_treatments]
        
        for t in range(num_treatments):
            mask = (test_dataset.treatments == t)
            if mask.any():
                t_pred = all_preds[mask, :, t]  # [batch_t, num_outcomes]
                t_true = test_dataset.outcomes[mask]
                t_mse = F.mse_loss(t_pred, t_true, reduction='none').mean(0)
                mse_matrix[t] = t_mse
                logger.info(f"Treatment {t} samples: {mask.sum().item()}")
        
        # Calculate CATE
        control_pred = all_preds[:, :, 0]  # [batch, num_outcomes]
        cate_dict = {}
        
        # Get true CATE from synthetic data
        synthetic_data = generate_data(
            N=cfg.data.num_samples,
            d=cfg.data.num_features,
            num_treatments=cfg.data.num_treatment,
            num_outcomes=len(cfg.data.outcome_cols),
            confounding_level=cfg.data.confounding_level,
            outcome_correlation=cfg.data.outcome_correlation,
            seed=cfg.seed
        )
        
        # Compare each treatment to control for each outcome
        test_indices = np.arange(len(test_dataset.features))  # Get indices for test set
        for m in range(num_outcomes):
            outcome_rmses = []
            for t in range(1, num_treatments):
                pred_cate = all_preds[:, m, t] - control_pred[:, m]
                true_cate = torch.FloatTensor(synthetic_data['true_cate'][f'outcome{m}'])[test_indices][:, t-1]
                rmse = torch.sqrt(((true_cate - pred_cate) ** 2).mean())
                cate_dict[f'outcome{m}_treatment{t}'] = rmse.item()
                outcome_rmses.append(rmse.item())
            cate_dict[f'outcome{m}_average'] = np.mean(outcome_rmses)
        
        # Calculate treatment-specific averages
        for t in range(1, num_treatments):
            cate_dict[f'treatment{t}_average'] = np.mean([
                cate_dict[f'outcome{m}_treatment{t}'] 
                for m in range(num_outcomes)
            ])
        
        # Calculate overall average
        cate_dict['average'] = np.mean([
            cate_dict[f'outcome{m}_average']
            for m in range(num_outcomes)
        ])

        results = {
            'train_loss': losses[0],
            'eval_loss': losses[1],
            'overall_mse': mse_matrix.mean().item(),
            'mse_matrix': mse_matrix.tolist(),
            'cate_rmse': cate_dict
        }

        # Print detailed results
        logger.info("\nDetailed MSE Analysis:")
        logger.info("Treatment | " + " | ".join([f"Outcome {i}" for i in range(num_outcomes)]))
        logger.info("-" * (10 + 12 * num_outcomes))
        for t in range(num_treatments):
            mse_values = [f"{mse:.4f}" for mse in mse_matrix[t]]
            logger.info(f"    {t}    | " + " | ".join(mse_values))
        
        logger.info("\nCATE RMSE Analysis:")
        header = "         |" + " | ".join([f" Treatment {t}" for t in range(1, num_treatments)]) + " | Average"
        logger.info(header)
        logger.info("-" * (len(header) + 5))
        
        for m in range(num_outcomes):
            rmse_values = [f"{cate_dict[f'outcome{m}_treatment{t}']:.4f}" for t in range(1, num_treatments)]
            logger.info(f"Outcome {m} | " + " | ".join(rmse_values) + f" | {cate_dict[f'outcome{m}_average']:.4f}")
        
        logger.info("-" * (len(header) + 5))
        avg_values = [f"{cate_dict[f'treatment{t}_average']:.4f}" for t in range(1, num_treatments)]
        logger.info(f"Average  | " + " | ".join(avg_values) + f" | {cate_dict['average']:.4f}")

    return results


if __name__ == "__main__":
    main()