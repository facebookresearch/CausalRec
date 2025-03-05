#!/bin/bash

#SBATCH --job-name=marginal_crn_gamma
#SBATCH --partition=q1       # change as needed, e.g., lowpri on some clusters
#SBATCH --gres=gpu:1        # uncomment only if/as needed
#SBATCH --time=2-00:00:00    # run for one day
#SBATCH --cpus-per-task=2    # change as needed
## %j is the job id, %u is the user id
#SBATCH --output=/data/home/yilingliu/CausalTransformer_avg/sbatch/marginal_crn_log/crn_marginal_gamma3-%j.log

cd /data/home/yilingliu/CausalTransformer_avg/
source venv/bin/activate
mlflow server --port=5005 &

PYTHONPATH=. python3 runnables/train_enc_dec.py \
    +dataset=cancer_sim \
    +backbone=crn \
    +backbone/crn_hparams/cancer_sim_marginal=\'3\' \
    exp.seed=10 \
    exp.mlflow_uri=http://127.0.0.1:5005

