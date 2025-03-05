#!/bin/bash

#SBATCH --job-name=criteo_tarnet
#SBATCH --partition=q1       # change as needed, e.g., lowpri on some clusters
#SBATCH --gres=gpu:1        # uncomment only if/as needed
#SBATCH --time=1-00:00:00    # run for one day
#SBATCH --cpus-per-task=2    # change as needed
## %j is the job id, %u is the user id
#SBATCH --output=/data/home/%u/MTML/sbatch/criteo_tarnet/criteo_tarnet-%j.log

cd /data/home/weishi0079/MTML
source activate mtml

python train.py