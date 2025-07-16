# 🎯 MOTTO: Mixture-of-Experts for Multi-Treatment Multi-Outcome Causal Inference

## ✨ Overview

**MOTTO** is a scalable framework for estimating treatment effects across **multiple treatments** and **multiple outcomes**, designed for applications like **advertising**, **healthcare**, and **recommendation systems**.

---

## 🧠 Why Multi-Treatment, Multi-Outcome?

- Real-world interventions are rarely binary.
- Outcomes are often correlated.
- Joint modeling improves data efficiency and generalization.

---

## 🔧 Architecture Highlights

- **Shared Expert Isolation**
  - Global-shared, treatment-shared, and outcome-shared experts
- **Selective Distribution Alignment (SDA)**
  - Balances latent treatment-shared representations across groups
- **Scalability**
  - Modular structure avoids combinatorial parameter growth

---

## 🚀 Key Features

- ✅ Expert routing tailored for treatment-outcome interactions
- ✅ Selective alignment avoids unnecessary distribution shifts
- ✅ Strong performance across both observational and randomized settings
- ✅ State-of-the-art results on synthetic and production datasets

---

## 📊 Evaluation Metrics

- **PEHE** – Precision in Estimating Heterogeneous Effects
- **AUUC** – Area Under Uplift Curve
- **AUCC** – Area Under Cost Curve

---


## 🧪 Model Variants

- `MOTTO` – Full model with SDA
- `MOTTO (no SDA)` – Ablation variant
- Baselines: Multi-task multi-treatment extension of `DragonNet`, `CFRNet`, `TARNet`, `S/T-learners`

---

## 🛠️ Usage

### 🧪 Synthetic Experiments

To run all baseline models on synthetic data:

```bash
cd demo
bash run_all_baselines.sh
```

This will execute all synthetic data experiments including:
- MOTTO (our proposed model)
- MOTTO_DA (with distribution alignment)
- Vanilla MMoE
- CFRNet (MMD and Wasserstein)
- TARNet
- DragonNet
- S-Learner and T-Learner

<!--
### 🔁 Single Run

```bash
python train.py
```

### 🔬 Hyperparameter Sweeping

```bash
python train.py -m hydra/sweeper/sampler=$sampler_choice objective=$metric
```
Where:
- `sampler_choice` can be: `random`, `tpe`, or `grid`
- `metric` can be: `auuc_outcome_0`, `auuc_outcome_1`, `auuc_outcome_no_trt1_0`, `auuc_outcome_no_trt1_1`
-->

## 📦 Requirements
- PyTorch
- Hydra
- TorchTNT
- TensorBoard
- Pandas
- Scikit-learn

## 📝 Citation

If you use this implementation in your research, please cite our work:
> ArXiv:
