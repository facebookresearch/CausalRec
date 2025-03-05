RTM & SGA Framework
==============================

A representation learning-based framework for counterfactual estimation in time series, introducing two novel techniques:
1. Sub-treatment Group Alignment (SGA) - Reduces confounding by aligning sub-treatment groups identified through Gaussian Mixture Models
2. Random Temporal Masking (RTM) - Improves causal information preservation by masking covariates with Gaussian noise at random timesteps

<img width="1518" alt="Architecture diagram showing RTM and SGA components" src="path_to_architecture_image">

## Key Features

- **Sub-treatment Group Alignment (SGA)**
  - Identifies sub-treatment groups using Gaussian Mixture Models (GMMs)
  - Performs distribution alignment between corresponding sub-groups
  - Theoretically and empirically proven to achieve better deconfounding
  
- **Random Temporal Masking (RTM)**
  - Masks covariates at random timesteps with Gaussian noise
  - Promotes selection of information crucial for future predictions
  - Reduces overfitting to factual outcomes
  - Preserves important causal information

## Project Structure

The project is built with following Python libraries:
1. [Pytorch-Lightning](https://pytorch-lightning.readthedocs.io/en/latest/) - deep learning models
2. [Hydra](https://hydra.cc/docs/intro/) - configuration management
3. [MlFlow](https://mlflow.org/) - experiment tracking

### Installation
```console
pip3 install virtualenv
python3 -m virtualenv -p python3 --always-copy venv
source venv/bin/activate
pip3 install -r requirements.txt
```

## Running Experiments

Main configuration file: `config/config.yaml`

Generic training script:
```console
PYTHONPATH=. CUDA_VISIBLE_DEVICES=<devices> 
python3 runnables/train.py +model=<model_type> +dataset=<dataset> exp.seed=<seed> exp.logging=True
```

### Model Types
- RTM only: `+model=rtm`
- SGA only: `+model=sga` 
- RTM+SGA combined: `+model=rtm_sga`

### Datasets
- Synthetic data: `+dataset=synthetic`
- Semi-synthetic MIMIC-III: `+dataset=mimic3_synthetic`

For MIMIC-III experiments, place [all_hourly_data.h5](https://github.com/MLforHealth/MIMIC_Extract) in `data/processed/`

Example running RTM+SGA on synthetic data with multiple seeds:
```console
PYTHONPATH=. CUDA_VISIBLE_DEVICES=<devices> 
python3 runnables/train.py +model=rtm_sga +dataset=synthetic exp.seed=10,101,1010,10101,101010
```

## Experiment Tracking

Start MLflow server:
```console
mlflow server --port=5000
```

Access web UI via SSH tunnel:
```console
ssh -N -f -L localhost:5000:localhost:5000 <username>@<server-link>
```

Then visit http://localhost:5000

## Results

Our framework achieves state-of-the-art performance on both synthetic and semi-synthetic datasets:

1. SGA demonstrates improved alignment between treatment groups compared to traditional distribution alignment methods
2. RTM shows better preservation of causal relationships and reduced overfitting
3. Combined RTM+SGA achieves the best performance in counterfactual outcome estimation

Detailed experimental results and ablation studies can be found in our paper.

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>.</small></p>
