#!/bin/bash

#SBATCH --job-name=subgroup_dynamic_weight_gamma
#SBATCH --partition=q1       # change as needed, e.g., lowpri on some clusters
#SBATCH --gres=gpu:1        # uncomment only if/as needed
#SBATCH --time=2-00:00:00    # run for one day
#SBATCH --cpus-per-task=2    # change as needed
## %j is the job id, %u is the user id
#SBATCH --output=/data/home/yilingliu/CausalTransformer/sbatch/subgroup_dynamic_weight_log/subgroupgamma4-%j.log

cd /data/home/yilingliu/CausalTransformer/
source venv/bin/activate
mlflow server --port=5000 &

PYTHONPATH=. python3 runnables/train_multi.py \
    +dataset=cancer_sim \
    +backbone=ct \
    +backbone/ct_hparams/cancer_sim_subgroup_dynamic_weight=\'4\' \
    exp.seed=10

