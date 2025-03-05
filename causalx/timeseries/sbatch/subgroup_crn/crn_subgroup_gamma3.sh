#!/bin/bash

#SBATCH --job-name=subgroup_crn_gamma
#SBATCH --partition=q1       # change as needed, e.g., lowpri on some clusters
#SBATCH --gres=gpu:1        # uncomment only if/as needed
#SBATCH --time=2-00:00:00    # run for one day
#SBATCH --cpus-per-task=2    # change as needed
## %j is the job id, %u is the user id
#SBATCH --output=/data/home/yilingliu/CausalTransformer_avg/sbatch/subgroup_crn_log/crn_subgroup_gamma3-%j.log

cd /data/home/yilingliu/CausalTransformer_avg/
source venv/bin/activate
mlflow server --port=5000 &

PYTHONPATH=. python3 runnables/train_enc_dec.py \
    +dataset=cancer_sim \
    +backbone=crn \
    +backbone/crn_hparams/cancer_sim_subgroup=\'3\' \
    exp.seed=10

