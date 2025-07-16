## **Adaptive Doubly Robust Learners for Long-term Treatment Effect Modeling**
An open-source variant of Ads Supply Personalization via Doubly Robust Learning [https://arxiv.org/abs/2410.12799]

### Instruction

#### Dataset Preparation
The demo dataset is Criteo Uplift Prediction Dataset: https://ailab.criteo.com/criteo-uplift-prediction-dataset/
1. Download the dataset from the provided link
2. Update the dataset path in `causalrec/lte/config/data/criteo.yaml` to point to your downloaded file location

#### Train the Doubly Robust Learners
To train a hybrid 2-stage doubly robust learner, please run:
```bash
python train.py
```

#### Hyperparameter Tuning
All hyperparameters can be configured in the `config/` directory.
