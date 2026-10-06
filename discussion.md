# Tabular Foundation Models (TFMs) - Intro, Dataset, and Starter!

Hi everyone, I want to share a **secret weapon** for tabular data problems. There exists pretrained backbones that already know how to predict tabular data **without** requiring any work from us. We **don't** need to engineer features, we **don't** need to tune hyperparameters, we **don't** need to train a model. 🎉

With **Tabular Foundation Models (TFMs)**, we just predict the labels of test data! Its so easy and so powerful! There is nothing for us to do! 🎉

# Frontier TFMs
TFMs have improved greatly in the past few months. The models below are the current SOTA and they are achieving these CV scores using `train.csv` as is with **no feature engineering**. That is amazing. These scores are beating GBDTs and MLPs that require hyperparameter tuning and feature engineering!

| Model | Precision | **OOF AUC** | Fit h (5 folds) | Predict h (5 folds) | Total GPU h | Peak GB |
|---|---|---:|---:|---:|---:|---:|
| tabpfn35 | amp | **0.961145** | 2.45 | 1.86 | 4.31 | 19.9 |
| kumolarge | fp16 | **0.960036** | 2.26 | 1.83 | 4.09 | 23.2 |
| **XGBoost (tuned)** | **BASELINE** | 0.959852 | 0.11 | 0.00 | 0.11 | 0.2 |
| **RealMLP (tuned)** | **BASELINE** | 0.959311 | 4.43 | 0.02 | 4.45 | 1.9 |
| tabfm | bf16 | **0.958871** | 0.03 | 7.46 | 7.49 | 45.2 |
| exaone | fp16 (exaone default) | **0.958557** | 0.00 | 10.71 | 10.71 | 75.0 |
| kumosmall | fp16 | **0.958484** | 0.54 | 0.46 | 1.00 | 25.1 |
| limix2 | fp16 autocast (limix default) | **0.958474** | 0.01 | 22.82 | 22.83 | 64.0 |
| causilo | fp16 (causilo policy) | **0.958275** | 0.04 | 0.91 | 0.95 | 31.1 |
| tabicl2 | fp16 autocast, column embedding fp32 | **0.958088** | 0.03 | 1.14 | 1.17 | 83.6 |
| mitra2 | fp16 autocast, logits fp32 | **0.957075** | 0.00 | 20.14 | 20.14 | 54.4 |

# Configuration
To provide the fair comparison above, each TFM uses `n_estimators=8`, `full context`, and `mixed precision`. So Opus5.5 had to rewrite many of the TFM libraries. Pilots were ran first to guarantee that code changes match the model's original accuracy. Also Opus5.5 added saving KV cache in each case to make inference very fast.

# Baselines - XGB and RealMLP 
We observe that **TabPFN3.5** and **Kumo-Tabular large** both beat XGBoost and RealMLP! This is amazing. The XGB and RealMLP have their hyperparameters optimized and use only the 21 raw features. They also use `n_estimators=8` similar to TFM.

# Kaggle Dataset
I publish a Kaggle dataset [here][1] with each of these TFM's OOF and Test PREDS. They were inferred offline using `StratifiedKFold(5, shuffle=True, random_state=42)` on my local A100 GPU.

# Starter Notebook
I publish a Kaggle starter notebook [here][2] demonstrating how to infer [**NVIDIA's Kumo-Tabular**][4] on this month's October playground competition data. And @philippsinger published a starter notebook [here][3] demonstrating how to infer [Prior Labs' TabPFN][5] on July playground competition data.

# Enjoy!

[1]: ???
[2]: ???
[3]: https://www.kaggle.com/code/philippsinger/tabpfn-3-starter-playground-series-s6e7
[4]: https://huggingface.co/nvidia/Kumo-Tabular
[5]: https://priorlabs.ai/tabpfn-3-5
