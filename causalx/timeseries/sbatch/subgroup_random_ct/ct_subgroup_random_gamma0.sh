#!/bin/bash

#SBATCH --job-name=subgroup_random_ct_gamma
#SBATCH --partition=q1       # change as needed, e.g., lowpri on some clusters
#SBATCH --gres=gpu:1        # uncomment only if/as needed
#SBATCH --time=4-00:00:00    # run for one day
#SBATCH --cpus-per-task=2    # change as needed
## %j is the job id, %u is the user id
#SBATCH --output=/data/home/yilingliu/CausalTransformer_avg/sbatch/subgroup_random_ct_log/subgroup_random_ct_gamma0-%j.log

cd /data/home/yilingliu/CausalTransformer_avg/
source venv/bin/activate
mlflow server --port=9008 &

PYTHONPATH=. python3 runnables/train_multi.py \
    +dataset=cancer_sim \
    +backbone=ct \
    +backbone/ct_hparams/cancer_sim_subgroup_random=\'0\' \
    exp.seed=10 \
    exp.mlflow_uri=http://127.0.0.1:9008


