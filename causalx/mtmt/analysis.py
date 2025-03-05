import logging
from datetime import datetime
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from typing import Dict
from dataset.dataset import UpliftDataset
from model.model import MLP, PLE2D, S_MMoE, Vanilla_MMoE, TARNet, Slearner, MultiTaskMoE, MultiTaskMoEDANN, MultiTaskMoETransformer
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
import matplotlib

logger = logging.getLogger()

def analyze_gate_patterns(gate_matrix):
        """
        Analyze gate patterns across tasks
        Args:
            gate_matrix: Tensor of shape [num_outcomes, num_treatments, num_experts]
            require_matrix: If True, return similarity matrix
        Returns:
            Dictionary of gate analysis metrics
        """
        analysis = {}
        
        # 1. Treatment-wise similarity within each outcome
        for outcome_idx in range(gate_matrix.shape[0]):
            treatment_gates = gate_matrix[outcome_idx]  # [5, num_experts]
            similarity = torch.corrcoef(treatment_gates)  # [5, 5]
            analysis[f'outcome_{outcome_idx}_treatment_similarity'] = similarity
            
        # 2. Outcome-wise similarity for each treatment
        for treatment_idx in range(gate_matrix.shape[1]):
            outcome_gates = gate_matrix[:, treatment_idx]  # [2, num_experts]
            similarity = torch.corrcoef(outcome_gates)  # [2, 2]
            analysis[f'treatment_{treatment_idx}_outcome_similarity'] = similarity
            
        # 3. Overall gate usage patterns
        task_indices = [(o, t) for o in range(gate_matrix.shape[0]) 
                               for t in range(gate_matrix.shape[1])]
        
        # Create full similarity matrix for all tasks
        all_gates = gate_matrix.reshape(-1, gate_matrix.shape[-1])  # [num_outcomes*num_treatments, num_experts]
        similarity_matrix = torch.corrcoef(all_gates)  # [num_tasks, num_tasks]
        
        # Add task indices for reference
        analysis['similarity_matrix'] = {
            'matrix': similarity_matrix,
            'task_indices': task_indices
        }
        
        # 4. Expert utilization per task
        expert_usage = gate_matrix.reshape(-1, gate_matrix.shape[-1]).mean(dim=0)  # [num_experts]
        analysis['expert_utilization'] = expert_usage
        
        return analysis

def visualize_gate_patterns(analysis):
    """
    Visualize gate patterns using matplotlib
    Args:
        analysis: Output from analyze_gate_patterns
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Plot full similarity matrix
    plt.figure(figsize=(12, 8))
    sim_matrix = analysis['similarity_matrix']['matrix']
    sns.heatmap(sim_matrix.cpu().numpy(), 
                xticklabels=[f'O{i//5}T{i%5}' for i in range(10)],
                yticklabels=[f'O{i//5}T{i%5}' for i in range(10)])
    plt.title('Gate Similarity Across Tasks')
    plt.savefig('gate_similarity.png')
    plt.close()
    
    # Plot expert utilization
    plt.figure(figsize=(10, 4))
    sns.barplot(x=list(range(len(analysis['expert_utilization']))), 
                y=analysis['expert_utilization'].cpu().numpy())
    plt.title('Expert Utilization Across Tasks')
    plt.xlabel('Expert Index')
    plt.ylabel('Average Gate Weight')
    plt.savefig('expert_utilization.png')
    plt.close()

    # Plot treatment-wise similarity for each outcome
    num_outcomes = len([k for k in analysis.keys() if k.startswith('outcome_')])
    for o in range(num_outcomes):
        plt.figure(figsize=(8, 6))
        sim = analysis[f'outcome_{o}_treatment_similarity']
        sns.heatmap(sim.cpu().numpy(),
                   xticklabels=[f'T{i}' for i in range(sim.shape[0])],
                   yticklabels=[f'T{i}' for i in range(sim.shape[0])])
        plt.title(f'Treatment Similarity for Outcome {o}')
        plt.savefig(f'treatment_similarity_outcome_{o}.png')
        plt.close()

    # Plot outcome-wise similarity for each treatment
    num_treatments = len([k for k in analysis.keys() if k.startswith('treatment_')])
    for t in range(num_treatments):
        plt.figure(figsize=(8, 6))
        sim = analysis[f'treatment_{t}_outcome_similarity']
        sns.heatmap(sim.cpu().numpy(),
                   xticklabels=[f'O{i}' for i in range(sim.shape[0])],
                   yticklabels=[f'O{i}' for i in range(sim.shape[0])])
        plt.title(f'Outcome Similarity for Treatment {t}')
        plt.savefig(f'outcome_similarity_treatment_{t}.png')
        plt.close()

# Example usage:
def get_gate_comparison_metrics(self, x):
    """
    Get specific comparison metrics between gates
    Returns dict with various similarity measures
    """
    analysis = self.analyze_gate_patterns(x)
    
    # Calculate specific metrics
    metrics = {}
    
    # 1. Average treatment similarity within outcomes
    treatment_sim = []
    for o in range(self.num_outcomes):
        sim = analysis[f'outcome_{o}_treatment_similarity']
        # Get upper triangle mean (excluding diagonal)
        mask = torch.triu(torch.ones_like(sim), diagonal=1).bool()
        treatment_sim.append(sim[mask].mean())
    metrics['avg_treatment_similarity'] = torch.stack(treatment_sim).mean()
    
    # 2. Average outcome similarity across treatments
    outcome_sim = []
    for t in range(self.num_treatments):
        sim = analysis[f'treatment_{t}_outcome_similarity']
        mask = torch.triu(torch.ones_like(sim), diagonal=1).bool()
        outcome_sim.append(sim[mask].mean())
    metrics['avg_outcome_similarity'] = torch.stack(outcome_sim).mean()
    
    # 3. Treatment clustering metric
    sim_matrix = analysis['similarity_matrix']['matrix']
    treatment_clusters = []
    for t in range(self.num_treatments):
        # Get indices for this treatment across outcomes
        indices = [t, t + self.num_treatments]
        cluster_sim = sim_matrix[indices][:, indices]
        treatment_clusters.append(cluster_sim.mean())
    metrics['treatment_clustering'] = torch.stack(treatment_clusters).mean()
    
    return metrics


def load_config_from_run(run_dir):
    config_path = os.path.join(run_dir, ".hydra/config.yaml")
    return OmegaConf.load(config_path)


@hydra.main(version_base=None, config_path="config", config_name="analysis_config")
def main(analysis_cfg: DictConfig):
    # Load config from previous run
    run_dir = analysis_cfg.run_dir
    cfg = load_config_from_run(run_dir)
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

    model.load_state_dict(torch.load(analysis_cfg.checkpoint, map_location=torch.device(device))["model_state_dict"])
    # Get full analysis
    x = test_dataset[:]
    gate_matrices = model.get_gate_matrices(x)
    print("gate_matrices", gate_matrices[0])
    analysis = analyze_gate_patterns(gate_matrices[0])
    print("analysis", analysis)
    visualize_gate_patterns(analysis)
    # print(analysis)
    # with matplotlib.pyplot.style.context('Agg'):
    # # Visualize patterns
    #     model.visualize_gate_patterns(analysis)

    # Get specific metrics
    # metrics = model.cgc_layers[0].get_gate_comparison_metrics(x)
    # print("Treatment Similarity:", metrics['avg_treatment_similarity'])
    # print("Outcome Similarity:", metrics['avg_outcome_similarity'])
    # print("Treatment Clustering:", metrics['treatment_clustering'])
    

if __name__ == "__main__":
    main()