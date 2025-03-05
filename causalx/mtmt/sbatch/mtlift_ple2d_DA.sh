#!/bin/bash

#SBATCH --job-name=mtlift_ple2d_da
#SBATCH --partition=q1       # change as needed, e.g., lowpri on some clusters
#SBATCH --gres=gpu:1        # uncomment only if/as needed
#SBATCH --time=2-00:00:00    # run for two days
#SBATCH --cpus-per-task=2    # change as needed
## %j is the job id, %u is the user id
#SBATCH --output=/data/home/%u/MTML/sbatch/mtlift_ple2d/mtlift_ple2d-%j.log
sampler_seed=$SLURM_ARRAY_TASK_ID
sampler_seed=$((sampler_seed))

conda activate mtml

python train.py -m hydra/sweeper/sampler=random \
    objective=auuc_outcome_0 \
    hydra.sweeper.n_trials=20 \
    hydra.sweeper.sampler.seed=$sampler_seed \
    model=ple2d_DA \
    seeds=[42,123,666]