# CausalRec: A Production-Ready Causal Learning Library for Industrial Recommendations
**CausalRec** is a python library developed by the applied causal learning team at Meta that provides State-of-the-Art causal modeling in industry recommendation systems, including a series of industry-level causal modeling solutions for different scenarios:
* **Multi-treatment Multi-outcome Causal Inference (KDD'25)**: introduces the first industry-level MoE framework (mixture-of-experts) with shared and dedicated experts for causal modeling, which addresses confounding biases and offers effective counterfactual modeling of cross-treatment, cross-outcome, treatment-outcome interactions through domain alignment.
* **Causal Transformers for Time-series Inference (Preprint)**: offers a novel framework for Transforemr models to effectively estimate counterfactual outcomes from time‑series observations that is crucial for effective decision-making, by synergistically integrating two complementary techniques for model improvement: Sub-treatment Group Alignment (SGA) uses iterative treatment‑agnostic clustering to identify fine-grained sub‑treatment groups, and Random Temporal Masking (RTM) promotes temporal generalization by randomly replacing input covariates with Gaussian noise during training.
* **Causality-driven Reinforcement Learning ([KDD'25](https://arxiv.org/abs/2501.05591))**: develops an offline dueling deep Q-network (DQN) based framework, which effectively mitigates confounding bias in dynamic systems and demonstrates significant offline gains, and further improves the framework's robustness against unanticipated distribution shifts.
* **Adaptive Doubly Robust Learners ([CIKM'24](https://arxiv.org/abs/2410.12799))**: introduces the first industry practice to adapt and apply doubly robust learners for long term causal effects modeling under the billion-scale industry applications, aiming to resolve the challenges in the billion-scale industry paradigm such as counterfactual modeling and weak treatment effects over an extended period of months.

# Getting Started

## Installation

```bash
pip install -e .
```

## Tutorials
We provide a series of tutorials to help you get started with CausalRec.

### 1. Multi-treatment Multi-outcome Causal Learning
A scalable framework called MOTTO is introduced for estimating treatment effects across multiple treatments and multiple outcomes, specifically designed for applications such as advertising, healthcare, and recommendation systems.

/motto_moe/demo/run_all_baselines.sh

### 2. Causal Transformer for Time-series Inference

Time series causal inference helps us understand how an intervention (e.g., treatment, policy, system change) influences outcomes that evolve over time.

/causal_time_series/runnables/train.py

### 3. Causality-driven Reinforcement Learning for Sequential Decision-Making

A real-time treatment estimation method based on an offline robust dueling Q-network is introduced, designed for session-level dynamic ad load optimization in advertising and recommendation systems.

/offline_robust_rl/train_rd3qn.py

### 4. Adaptive Doubly Robust Learner for Long-term Causal Effects Estimation

A long-term treatment estimation method via doubly robust learning is introduced, designed for user-level ad load optimization in advertising and recommendation systems.

/adaptive_drl/train.py

# License

CausalRec is released under the MIT License. See the LICENSE file for more details.
