#!/bin/bash

#SBATCH --job-name=marginal_gamma
#SBATCH --partition=q1       # change as needed, e.g., lowpri on some clusters
#SBATCH --gres=gpu:1        # uncomment only if/as needed
#SBATCH --time=1-00:00:00    # run for one day
#SBATCH --cpus-per-task=2    # change as needed
## %j is the job id, %u is the user id
#SBATCH --output=/data/home/yilingliu/CausalTransformer/sbatch/marginal_log/marginalgamma2-%j.log

cd /data/home/yilingliu/CausalTransformer/
source venv/bin/activate
mlflow server --port=5000 &

PYTHONPATH=. python3 runnables/train_multi.py \
    +dataset=cancer_sim \
    +backbone=ct \
    +backbone/ct_hparams/cancer_sim_marginal=\'2\' \
    exp.seed=10 \
    model.multi.optimizer.learning_rate=0.02 \
    model.multi.batch_size=1024 \
    model.multi.seq_hidden_units=16 \
    model.multi.dropout_rate=0.1 \
    exp.alpha_wass=0.0001 \
    exp.alpha_wass_epoch=False \
    exp.alpha_wass_growth=0.001 \
    exp.alpha_wass_max=0.01 \
    exp.max_epochs=200 


PYTHONPATH=. python3 runnables/train_multi.py \
    +dataset=cancer_sim \
    +backbone=ct \
    +backbone/ct_hparams/cancer_sim_marginal=\'2\' \
    exp.seed=10 \
    model.multi.optimizer.learning_rate=0.02 \
    model.multi.batch_size=2048 \
    model.multi.seq_hidden_units=16 \
    model.multi.dropout_rate=0.1 \
    exp.alpha_wass=0.0001 \
    exp.alpha_wass_epoch=False \
    exp.alpha_wass_growth=0.001 \
    exp.alpha_wass_max=0.01 \
    exp.max_epochs=200 

PYTHONPATH=. python3 runnables/train_multi.py \
    +dataset=cancer_sim \
    +backbone=ct \
    +backbone/ct_hparams/cancer_sim_marginal=\'2\' \
    exp.seed=10 \
    model.multi.optimizer.learning_rate=0.02 \
    model.multi.batch_size=2048 \
    model.multi.seq_hidden_units=16 \
    model.multi.dropout_rate=0.1 \
    exp.alpha_wass=0.0001 \
    exp.alpha_wass_epoch=False \
    exp.alpha_wass_growth=0.001 \
    exp.alpha_wass_max=0.01 \
    exp.max_epochs=100 

PYTHONPATH=. python3 runnables/train_multi.py \
    +dataset=cancer_sim \
    +backbone=ct \
    +backbone/ct_hparams/cancer_sim_marginal=\'2\' \
    exp.seed=10 \
    model.multi.optimizer.learning_rate=0.02 \
    model.multi.batch_size=2048 \
    model.multi.seq_hidden_units=16 \
    model.multi.dropout_rate=0.1 \
    exp.alpha_wass=0.0001 \
    exp.alpha_wass_epoch=False \
    exp.alpha_wass_growth=0.001 \
    exp.alpha_wass_max=0.01 \
    exp.max_epochs=150 

PYTHONPATH=. python3 runnables/train_multi.py \
    +dataset=cancer_sim \
    +backbone=ct \
    +backbone/ct_hparams/cancer_sim_marginal=\'2\' \
    exp.seed=10 \
    model.multi.optimizer.learning_rate=0.01 \
    model.multi.batch_size=2048 \
    model.multi.seq_hidden_units=16 \
    model.multi.dropout_rate=0.1 \
    exp.alpha_wass=0.0001 \
    exp.alpha_wass_epoch=False \
    exp.alpha_wass_growth=0.001 \
    exp.alpha_wass_max=0.01 \
    exp.max_epochs=200


PYTHONPATH=. python3 runnables/train_multi.py \
    +dataset=cancer_sim \
    +backbone=ct \
    +backbone/ct_hparams/cancer_sim_marginal=\'2\' \
    exp.seed=10 \
    model.multi.optimizer.learning_rate=0.03 \
    model.multi.batch_size=2048 \
    model.multi.seq_hidden_units=16 \
    model.multi.dropout_rate=0.1 \
    exp.alpha_wass=0.0001 \
    exp.alpha_wass_epoch=False \
    exp.alpha_wass_growth=0.001 \
    exp.alpha_wass_max=0.01 \
    exp.max_epochs=200




PYTHONPATH=. python3 runnables/train_multi.py \
    +dataset=cancer_sim \
    +backbone=ct \
    +backbone/ct_hparams/cancer_sim_marginal=\'2\' \
    exp.seed=10 \
    model.multi.optimizer.learning_rate=0.02 \
    model.multi.batch_size=1024 \
    model.multi.seq_hidden_units=16 \
    model.multi.dropout_rate=0.1 \
    exp.alpha_wass=0.0001 \
    exp.alpha_wass_epoch=True \
    exp.alpha_wass_growth=0.001 \
    exp.alpha_wass_max=0.01 \
    exp.max_epochs=200 


PYTHONPATH=. python3 runnables/train_multi.py \
    +dataset=cancer_sim \
    +backbone=ct \
    +backbone/ct_hparams/cancer_sim_marginal=\'2\' \
    exp.seed=10 \
    model.multi.optimizer.learning_rate=0.02 \
    model.multi.batch_size=2048 \
    model.multi.seq_hidden_units=16 \
    model.multi.dropout_rate=0.1 \
    exp.alpha_wass=0.0001 \
    exp.alpha_wass_epoch=True \
    exp.alpha_wass_growth=0.001 \
    exp.alpha_wass_max=0.01 \
    exp.max_epochs=200 

PYTHONPATH=. python3 runnables/train_multi.py \
    +dataset=cancer_sim \
    +backbone=ct \
    +backbone/ct_hparams/cancer_sim_marginal=\'2\' \
    exp.seed=10 \
    model.multi.optimizer.learning_rate=0.02 \
    model.multi.batch_size=2048 \
    model.multi.seq_hidden_units=16 \
    model.multi.dropout_rate=0.1 \
    exp.alpha_wass=0.0001 \
    exp.alpha_wass_epoch=True \
    exp.alpha_wass_growth=0.001 \
    exp.alpha_wass_max=0.01 \
    exp.max_epochs=100 

PYTHONPATH=. python3 runnables/train_multi.py \
    +dataset=cancer_sim \
    +backbone=ct \
    +backbone/ct_hparams/cancer_sim_marginal=\'2\' \
    exp.seed=10 \
    model.multi.optimizer.learning_rate=0.02 \
    model.multi.batch_size=2048 \
    model.multi.seq_hidden_units=16 \
    model.multi.dropout_rate=0.1 \
    exp.alpha_wass=0.0001 \
    exp.alpha_wass_epoch=True \
    exp.alpha_wass_growth=0.001 \
    exp.alpha_wass_max=0.01 \
    exp.max_epochs=150 

PYTHONPATH=. python3 runnables/train_multi.py \
    +dataset=cancer_sim \
    +backbone=ct \
    +backbone/ct_hparams/cancer_sim_marginal=\'2\' \
    exp.seed=10 \
    model.multi.optimizer.learning_rate=0.01 \
    model.multi.batch_size=2048 \
    model.multi.seq_hidden_units=16 \
    model.multi.dropout_rate=0.1 \
    exp.alpha_wass=0.0001 \
    exp.alpha_wass_epoch=True \
    exp.alpha_wass_growth=0.001 \
    exp.alpha_wass_max=0.01 \
    exp.max_epochs=200


PYTHONPATH=. python3 runnables/train_multi.py \
    +dataset=cancer_sim \
    +backbone=ct \
    +backbone/ct_hparams/cancer_sim_marginal=\'2\' \
    exp.seed=10 \
    model.multi.optimizer.learning_rate=0.03 \
    model.multi.batch_size=2048 \
    model.multi.seq_hidden_units=16 \
    model.multi.dropout_rate=0.1 \
    exp.alpha_wass=0.0001 \
    exp.alpha_wass_epoch=True \
    exp.alpha_wass_growth=0.001 \
    exp.alpha_wass_max=0.01 \
    exp.max_epochs=200
















