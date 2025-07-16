# ⏳ Time Series Causal Inference with SGA + RTM

## 🧠 What is Time Series Causal Inference?

**Time series causal inference** helps us understand how an intervention (e.g., treatment, policy, system change) influences outcomes that evolve **over time**.

Unlike prediction tasks, the goal is to answer **“what would have happened if we had done something differently — at a different time?”**

---

## 🔍 Real-World Use Cases

- **🏥 Healthcare**  
  > When is the best time to administer treatment?  
  E.g., What if we delayed surgery by two days for a patient with DCIS?

- **📊 Economics**  
  > How does a policy affect GDP growth over months?  
  E.g., Would the economy have recovered faster *without* a stimulus?

- **🏭 Manufacturing**  
  > What’s the long-term effect of changing machine settings?  
  E.g., Does increasing pressure improve output, or hurt long-term product quality?

- **📱 Online Experiments**  
  > What if we didn’t show the new version of the app?  
  E.g., How does a UI change today affect user engagement over the next week?

---

## 🧩 Why It’s Hard

- **Time-varying confounding**  
  Factors that affect both treatment and outcome can **change over time**.

- **Sequential dependency**  
  Today’s decisions affect tomorrow’s options.

- **Missing counterfactuals**  
  We only observe what *did* happen — not what *could* have happened.

---

## ✨ Our Contribution: SGA + RTM

We propose a general-purpose framework for time series causal inference based on **representation learning**, with two novel techniques:

### 🔄 Sub-treatment Group Alignment (SGA)
- Instead of just aligning treatment and control groups...
- We identify **subgroups** (e.g., by demographics or health profiles) using **Gaussian Mixture Models**,  
  then align these subgroups **across treatment arms**.
- This leads to **finer-grained deconfounding** and **tighter error bounds**.

### 🕳️ Random Temporal Masking (RTM)
- Inspired by masked language modeling (like BERT),
- We randomly **mask covariates at certain time points** using Gaussian noise during training.
- This forces the model to rely on **stable causal relationships across time** — not just recent trends.
- Helps **reduce overfitting** and **prevents error accumulation** over long sequences.

---


## 🧱 Project Structure

This project is built using:

- [PyTorch Lightning](https://pytorch-lightning.readthedocs.io/en/latest/) — deep learning
- [Hydra](https://hydra.cc/docs/intro/) — config management
- [MLflow](https://mlflow.org/) — experiment tracking

---

## ⚙️ Installation

```bash
pip3 install virtualenv
python3 -m virtualenv -p python3 --always-copy venv
source venv/bin/activate
pip3 install -r requirements.txt
```
---

## 🚀 Running Experiments

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

---

## 📊 Experiment Tracking with MLflow

Start MLflow server:
```console
mlflow server --port=5000
```

Access web UI via SSH tunnel:
```console
ssh -N -f -L localhost:5000:localhost:5000 <username>@<server-link>
```

Then visit http://localhost:5000

---

## 📈 Performance Highlights

- 📊 State-of-the-art performance on both **synthetic** and **semi-synthetic** datasets  
- 💪 Particularly effective in **high-confounding** scenarios  
- 🧩 Robust to the number of clusters and model choice (RNNs or Transformers)
- 🕶 This framework is model-agnostic and can be plugged into any sequence model (e.g., CRN, CT). 
- 📜 Detailed experimental results and ablation studies can be found in our paper.

---

## 📝 Citation
If you use this implementation in your research, please cite our work:
> ArXiv:

---

## 🙌 Acknowledgement

This codebase is built on top of [CausalTransformer](https://github.com/Valentyn1997/CausalTransformer).



