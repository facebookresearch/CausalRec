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


parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
from dataset.dataset import UpliftDataset
from model.model import Slearner, TARNet
import pdb

logger = logging.getLogger()


# def predict(model, test_dataset, outcome_index, trt_index, model_type="t"):
#     """
#     Multi-task, multi-treatment prediction

#     Parameters:
#     - model: Slearner/TARNet/...
#     - test_dataset: UpliftDataset
#     - outcome_index: List[int]
#     - trt_index: int
#     - model_type: str, s for slearner; t for tlearner

#     Returns:
#     - pred: tensor [N,]
#     """
#     model.eval()
#     if model_type == "s":
#         test_dataset.assign_trt(trt_index)
#     samples = test_dataset[:]
#     with torch.no_grad():
#         if model_type == "adversarial":
#             pred, _ = model(samples) # [batch x n_outcomes x m_treatments]
#         elif model_type == "transformer":
#             print(f"Batch size for evaluation: {samples['feature'].shape[0]}")
#             print("Samples keys:", samples.keys() if isinstance(samples, dict) else type(samples))
#             pred = model(samples) 
#         else:
#             pred = model(samples) # [batch x n_outcomes x m_treatments]
#     pred = pred[:, outcome_index, :] # [batch x m_treatments]
#     if model_type == "t" or model_type == "adversarial":
#         pred = pred[:, trt_index] # [batch]
#     else:
#         pred = pred[:, 0] # [batch]
#     return pred

def predict(model, test_dataset, outcome_index, trt_index, model_type="t"):
    """
    Multi-task, multi-treatment prediction

    Parameters:
    - model: Slearner/TARNet/...
    - test_dataset: UpliftDataset
    - outcome_index: List[int] or int
    - trt_index: int
    - model_type: str, 's' for slearner; 't' for tlearner; 'adversarial' or 'transformer'

    Returns:
    - pred: tensor [N,] - predictions for the specified outcome and treatment index
    """
    model.eval()  # Set the model to evaluation mode

    # For s-learner, modify the dataset to assign a specific treatment index
    if model_type == "s":
        test_dataset.assign_trt(trt_index)
    
    # Ensure consistent batch processing, particularly for large datasets
    batch_size = 16384 # Set a reasonable batch size for evaluation
    eval_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    predictions = []
    with torch.no_grad():  # Disable gradient computation during inference
        for samples in eval_loader:
            # Handle different model types specifically
            if model_type == "adversarial":
                pred, _ = model(samples)  # Assuming adversarial model returns extra outputs
            else:
                pred = model(samples)  # Standard prediction for other model types
            
            # Extract predictions for the specific outcome and treatment index
            pred = pred[:, outcome_index, :]  # Select the specific outcome
            if not isinstance(outcome_index, list):
                if model_type in ["t", "adversarial", "transformer"]:
                    pred = pred[:, trt_index]  # Select the specific treatment index
                else:
                    pred = pred[:, 0]  # Default to the first treatment if unspecified
            else:
                if model_type in ["t", "adversarial", "transformer"]:
                    pred = pred[:, :, trt_index]  # Select the specific treatment index
                else:
                    pred = pred[:, :, 0]  # Default to the first treatment if unspecified

            predictions.append(pred)

    # Concatenate all batch predictions to form a complete array
    predictions = torch.cat(predictions, dim=0)
    return predictions

def evaluate_auuc(uplift_scores_all, A_test, Y_test, treatments, outcome_index, metric='average', plot=True, save_path=None):
    """
    Evaluate the multi-treatment AUUC

    Parameters:
    - uplift_scores_all (np.ndarray): Uplift scores for all treatments [N, A].
    - A_test (np.ndarray): Actual treatments assigned to the test samples. [N, 1]
    - Y_test (np.ndarray): Actual outcomes for the primary task. [N, 1]
    - treatments (list): List of treatment identifiers for which the models will predict outcomes. [A]
    - metric (str): Type of AUUC calculation to perform ('average' or 'optimal').
    - plot (bool): plot the gain chart.

    Returns:
    - auuc_results (dict): If optimal_auuc, overall AUUC scores.
    - auuc_results (dict): If average, a dictionary containing the AUUC scores for each treatment.
    """
    
    auuc_results = {}
    if metric == 'optimal':
        # Select the treatment with the highest uplift score
        max_uplift_indices = np.argmax(uplift_scores_all, axis=1)
        max_uplift_scores = uplift_scores_all[np.arange(uplift_scores_all.shape[0]), max_uplift_indices]
        recommended_treatments = np.array(treatments)[max_uplift_indices]

        data = {
            'treatment': A_test.ravel(),
            'outcome': Y_test.ravel(),
            'uplift_score': max_uplift_scores,
            'recommended_treatment': recommended_treatments
        }
        df = pd.DataFrame(data)
        df_filtered = df[(df['treatment'] == 0) | (df['treatment'] == df['recommended_treatment'])].copy()
        df_filtered['binary_treatment'] = (df_filtered['treatment'] == df_filtered['recommended_treatment']).astype(int)
        df_filtered_col = df_filtered[['outcome', 'binary_treatment', 'uplift_score']]
       
        auuc = auuc_score(
            df_filtered_col,
            outcome_col='outcome',
            treatment_col='binary_treatment',
            normalize=True
        )
        optimal_auuc = auuc["uplift_score"]
        
        logger.info(f"Optimal AUUC across all treatments: {optimal_auuc}")
        
        auuc_results["optimal"] = optimal_auuc
        if plot:
            plt.figure(figsize=(8, 6))
            plot_gain(
                df_filtered_col,
                outcome_col='outcome',
                treatment_col='binary_treatment',
                normalize=True,
                figsize=(8, 6),
            )
            plt.title(f'Gain Chart for Optimal Treatment')
            
            # Save the plot 
            if save_path is None:
                save_path_final = os.path.join(os.getcwd(), 'gain_chart_outcome'+str(outcome_index)+'_treatment_'+str(t)+'.png')
            else:
                save_path_final = os.path.join(save_path, 'gain_chart_outcome'+str(outcome_index)+'_treatment_'+str(t)+'.png')
            
            plt.savefig(save_path_final,dpi=150, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path_final}")
            
            plt.show()

        #optimal_auuc is only for placeholder for keep the output format consistent
        return optimal_auuc, auuc_results, optimal_auuc

    elif metric == 'average':
        # Calculate average uplift scores
        for idx_t, t in enumerate(treatments[1:]):
            data = {
                'treatment': A_test.ravel(),
                'outcome': Y_test.ravel(),
                'uplift_score': uplift_scores_all[:, idx_t],
            }
            df = pd.DataFrame(data)
            df_filtered = df[(df['treatment'] == 0) | (df['treatment'] == t)].copy()
            df_filtered['binary_treatment'] = (df_filtered['treatment'] == t).astype(int)
            df_filtered_col = df_filtered[['outcome', 'binary_treatment', 'uplift_score']]
            auuc = auuc_score(
                df_filtered_col,
                outcome_col='outcome',
                treatment_col='binary_treatment',
                normalize=True
            )
            auuc_results[str(t)] = auuc["uplift_score"]

            if plot:
                plt.figure(figsize=(8, 6))
                plot_gain(
                    df_filtered_col,
                    outcome_col='outcome',
                    treatment_col='binary_treatment',
                    normalize=True,
                    figsize=(8, 6),
                )
                plt.title(f'Gain Chart for Treatment {t}')
                
                 # Save the plot 
                if save_path is None:
                    save_path_final = os.path.join(os.getcwd(), 'gain_chart_outcome'+str(outcome_index)+'_treatment_'+str(t)+'.png')
                else:
                    save_path_final = os.path.join(save_path, 'gain_chart_outcome'+str(outcome_index)+'_treatment_'+str(t)+'.png')
            
                plt.savefig(save_path_final, dpi=150, bbox_inches='tight')
                logger.info(f"Plot saved to {save_path_final}")
                plt.show()

        average_auuc = np.mean(list(auuc_results.values()))
        
        # Exclude treatment 1 and compute the average AUUC
        average_auuc_exclude_1  = np.mean(list(auuc_results.values())[1:])
        
        logger.info(f"Average AUUC across all treatments: {average_auuc}")

    return average_auuc, auuc_results, average_auuc_exclude_1


def eval_multi_trt(model, test_dataset, outcome_index, trt_list, metric_type="average", model_type="t", plot=False, save_path=None):
    preds_cate = []
    base_trt = trt_list[0] # assume the 1st trt is the control version
    
    logger.info(f"control_version: {base_trt}")
    
    pred_ctrl = predict(model, test_dataset, outcome_index, base_trt, model_type)
    for trt_index in trt_list[1:]:
        pred_trt = predict(model, test_dataset, outcome_index, trt_index, model_type)
        pred_cate = pred_trt - pred_ctrl
        preds_cate.append(pred_cate)
    uplift_scores_all = torch.stack(preds_cate, dim=1).cpu().numpy() # [batch_size, num_trt - 1]
    A_test = test_dataset.get_trt()
    Y_test = test_dataset.get_outcome()[:, outcome_index]
    auuc, auuc_results, average_auuc_exclude_1 = evaluate_auuc(uplift_scores_all, A_test, Y_test, trt_list, outcome_index, metric_type, plot, save_path)
    return auuc, auuc_results, average_auuc_exclude_1


def eval_multi_trt_ctcvr(model, test_dataset, outcome_index, trt_list, metric_type="average", model_type="t", plot=False, save_path=None):
    preds_cate = []
    base_trt = trt_list[0] # assume the 1st trt is the control version
    
    logger.info(f"control_version: {base_trt}")
    
    pred_ctrl = predict(model, test_dataset, [0, 1], base_trt, model_type)
    pred_ctrl = pred_ctrl[:, 0] * pred_ctrl[:, 1] # handle the ctcvr case
    for trt_index in trt_list[1:]:
        pred_trt = predict(model, test_dataset, [0, 1], trt_index, model_type)
        pred_trt = pred_trt[:, 0] * pred_trt[:, 1] # handle the ctcvr case        
        pred_cate = pred_trt - pred_ctrl
        preds_cate.append(pred_cate)
    uplift_scores_all = torch.stack(preds_cate, dim=1).cpu().numpy() # [batch_size, num_trt - 1]
    A_test = test_dataset.get_trt()
    Y_test = test_dataset.get_outcome()[:, outcome_index]
    auuc, auuc_results, average_auuc_exclude_1 = evaluate_auuc(uplift_scores_all, A_test, Y_test, trt_list, outcome_index, metric_type, plot, save_path)
    return auuc, auuc_results, average_auuc_exclude_1
    

def calculate_stats(data_list):
    averages = {}
    std_devs = {}
    keys = data_list[0].keys()
    for key in keys:
        first_value = data_list[0][key]
        
        if isinstance(first_value, dict):
            # Recursively calculate stats for nested dictionaries
            nested_data = [d[key] for d in data_list]
            nested_averages, nested_std_devs = calculate_stats(nested_data)
            averages[key] = nested_averages
            std_devs[key] = nested_std_devs
        elif isinstance(first_value, (int, float)):
            values = np.array([d[key] for d in data_list])
            avg = np.mean(values)
            std_dev = np.std(values, ddof=1)  # Use ddof=1 for sample standard deviation
            
            averages[key] = avg
            std_devs[key] = std_dev
        else:
            raise ValueError("Unsupported data type. Expected float or dict.")
    return averages, std_devs


class TestUtils():
    def test_optimal_metric(self, num_trt):
        uplift_scores_all = np.random.rand(100, 3)
        A_test = np.random.randint(0, num_trt, size=(100, 1))
        Y_test = np.random.randint(0, 2, size=(100, 1))
        treatments = [0, 1, 2]
        
        optimal_auuc, auuc_results = evaluate_auuc(uplift_scores_all, A_test, Y_test, treatments, outcome_index=0, metric='optimal', plot=False)
       
        print(auuc_results)
        
    def test_average_metric(self, num_trt):
        uplift_scores_all = np.random.rand(100, 3)
        A_test = np.random.randint(0, num_trt, size=(100, 1))
        Y_test = np.random.randint(0, 2, size=(100, 1))
        treatments = [0, 1, 2]
        
        average_auuc, auuc_results = evaluate_auuc(uplift_scores_all, A_test, Y_test, treatments, outcome_index=0, metric='average', plot=False)
        
        print(auuc_results)

    def gen_test_data(self, num_trt, num_sample=10):
        config = {
            "feature_cols": ["X1", "X2"],
            "treatment_col": "A",
            "outcome_cols": ["O1", "O2", "O3"]
        }
        np.random.seed(42)
        X1 = np.random.rand(num_sample).astype(float)
        X2 = np.random.rand(num_sample).astype(float)
        A = np.random.randint(0, num_trt, size=num_sample).astype(int)
        O1 = np.random.rand(num_sample).astype(float)
        O2 = np.random.rand(num_sample).astype(float)
        O3 = np.random.rand(num_sample).astype(float)
        df = pd.DataFrame({
            "X1": X1,
            "X2": X2,
            "A": A,
            "O1": O1,
            "O2": O2,
            "O3": O3
        })
        device = init_from_env()
        test_dataset = UpliftDataset(df, config, device)
        return test_dataset

    def test_model_inference(self):
        test_dataset = self.gen_test_data(num_trt=2)
        tarnet = TARNet(input_dim=2, num_outcome=3, num_treatment=2)
        slearner = Slearner(input_dim=2, num_outcome=3)
        for trt in [0, 1]:
            for outcome in [0, 1, 2]:
                for model in [tarnet, slearner]:
                    if isinstance(model, TARNet):
                        model_type = "t"
                    else:
                        model_type = "s"
                    pred = predict(model, test_dataset, outcome, trt, model_type)
                    print(pred.shape)
                    
    def test_eval_multi_trt(self, num_trt, num_sample):
        test_dataset = self.gen_test_data(num_trt, num_sample)
        tarnet = TARNet(input_dim=2, num_outcome=3, num_treatment=num_trt)
        slearner = Slearner(input_dim=2, num_outcome=3)
        model_types = ["s", "t"]
        trt_list = np.arange(num_trt)
        for outcome_index in [0, 1, 2]:
            for i, model in enumerate([slearner, tarnet]):
                model_type = model_types[i]
                auuc, auuc_results = eval_multi_trt(model, test_dataset, outcome_index, trt_list, "average", model_type, False)
                print("end auuc:", auuc)
                print("all auucs:", auuc_results)
                
    def test_calculate_stats(self):
        data = [
            {'a': 1, 'b': {'x': 4, 'y': 6}, 'c': 7},
            {'a': 1, 'b': {'x': 5, 'y': 7}, 'c': 9},
            {'a': 2, 'b': {'x': 7, 'y': 9}, 'c': 11}
        ]
        mean_dict, std_dict = calculate_stats(data)
        print("mean_dict:", mean_dict)
        print("std_dict:", std_dict)
                
        
def get_da_loss(method='mmd', blur=0.05, scaling=0.95):
    """
    Get domain adaptation loss function (MMD or Wasserstein)
    Uses the geomloss package for efficient computation
    
    Args:
        method: 'mmd' or 'wasserstein'
        blur: blur parameter for the Sinkhorn divergence
        scaling: scaling parameter for the Sinkhorn divergence
    
    Returns:
        loss_fn: geomloss.SamplesLoss function
    """
    if method == 'mmd':
        # For MMD, use Gaussian kernel via Energy Distance
        return SamplesLoss(
            loss="energy",
            backend="tensorized"
        )
    elif method == 'wasserstein':
        # For Wasserstein, use Sinkhorn divergence
        return SamplesLoss(
            loss="sinkhorn",
            p=1,
            blur=blur,
            scaling=scaling,
            backend="tensorized"
        )
    else:
        raise ValueError(f"Unknown method: {method}. Choose 'mmd' or 'wasserstein'")

if __name__ == '__main__':
    test = TestUtils()
    # test.test_average_metric(2)
    # test.test_optimal_metric(2)
    # test.test_model_inference()
    # test.test_eval_multi_trt(3, 100)
    test.test_calculate_stats()

