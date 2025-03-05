#!/bin/bash

#SBATCH --job-name=semi_crn_marginal
#SBATCH --partition=q1       # change as needed, e.g., lowpri on some clusters
#SBATCH --gres=gpu:1        # uncomment only if/as needed
#SBATCH --time=3-00:00:00    # run for one day
#SBATCH --cpus-per-task=2    # change as needed
## %j is the job id, %u is the user id
#SBATCH --output=/data/home/yilingliu/CausalTransformer_avg/sbatch/semi_synthetic_crn_log/crn_marginal-%j.log

cd /data/home/yilingliu/CausalTransformer_avg/
source venv/bin/activate
mlflow server --port=6002 &

PYTHONPATH=. python3 runnables/train_enc_dec.py \
    +dataset=mimic3_synthetic \
    +backbone=crn \
    +backbone/crn_hparams/mimic3_synthetic=1000_marginal.yaml \
    exp.seed=10 \
    exp.mlflow_uri=http://127.0.0.1:6002

