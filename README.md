<div align="center">

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1-red?logo=pytorch)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange?logo=tensorflow)
![CUDA](https://img.shields.io/badge/CUDA-11.8-success?logo=nvidia)
![MLflow](https://img.shields.io/badge/MLflow-Enabled-blueviolet?logo=mlflow)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker)
![Transformers](https://img.shields.io/badge/Transformers-Attention--Based-brightgreen)
![SequenceModeling](https://img.shields.io/badge/Sequence--Modeling-RNN%2FLSTM%2FTransformer-brightgreen)
![RecSys](https://img.shields.io/badge/RecSys-Next--Item--Prediction-yellow)
![GPU Training](https://img.shields.io/badge/Training-GPU%20Accelerated-blue)
![Build](https://img.shields.io/badge/Build-Passing-brightgreen)
![Research](https://img.shields.io/badge/Research-Grade-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

</div>

 Project Summary

DeepSequence-Recommender is a sequence-aware recommendation system that predicts the next item a user will interact with, using modern deep learning architectures:

GRU4Rec (Gated Recurrent Unit)

LSTM (Long Short-Term Memory)

Transformer (SASRec-style) with attention + positional embeddings


This repo is structured and documented to match FAANG-level ML Engineering + Research Lab expectations.

Included documentation:

📊 metrics.md – All evaluation metrics

🧪 ablation_study.md – Architectural comparisons

📄 model_card.md – Standardized model documentation

🏁 benchmark.md – Baseline vs model comparisons

📊 dataset_stats.md – Dataset analysis and sequence distributions

🔁 reproducibility.md – Instructions for deterministic reproduction


 Key Features

✔ Multi-architecture support: GRU, LSTM, Transformer

✔ Research-grade evaluation metrics for RecSys

✔ NDCG@K, Precision@K, Recall@K, Hit Rate, MRR

✔ Deep sequence modeling (positional encodings, attention layers)

✔ Deterministic pipeline with fixed seeds

✔ MLflow experiment logging

✔ Docker-ready structure

✔ Modular, readable code in src/ and scripts/


Repository Structure

DeepSequence-Recommender/
│
├── src/
│   ├── models/
│   ├── trainer.py
│   ├── evaluator.py
│   └── utils.py
│
├── scripts/
│   ├── train.py
│   └── evaluate.py
│
├── metrics.md
├── ablation_study.md
├── model_card.md
├── benchmark.md
├── dataset_stats.md
├── reproducibility.md
│
├── requirements.txt
└── README.md


Research Metrics (Highlights)

From metrics.md:

Metric	Value

Precision@10	0.271
Recall@10	0.334
NDCG@10	0.357
Hit Rate@10	0.612
MRR	0.421


🔗 Full results: metrics.md

 Ablation Study (Highlights)

From ablation_study.md:

Variant	NDCG@10	Notes

Transformer (2-layer)	0.357	Best model
LSTM	0.346	Strong
GRU	0.328	Good baseline
No Positional Encoding	0.301	Large drop
Short Seq Len	0.287	Not enough context


🔗 Full study: ablation_study.md

 Benchmark Comparison

From benchmark.md:

Model	NDCG@10	Hit Rate

Transformer (2-layer)	0.357	0.612
LSTM	0.346	0.628
GRU	0.328	0.612
ItemKNN	0.192	0.401
Popularity	0.086	0.215


🔗 Full comparison: benchmark.md


Dataset Statistics (Highlights)

From dataset_stats.md:

Sequence Length	% of Users

1–5	42%
6–15	33%
16–30	19%
31–50	6%


Dataset is sparse → sequence models are necessary.

🔗 Full stats: dataset_stats.md


Reproducibility

Run deterministic training:

python train.py --model transformer --epochs 20 --seed 42

Evaluate:

python evaluate.py --model transformer --topk 10

🔗 Full guide: reproducibility.md



 Installation

git clone https://github.com/Trojan3877/DeepSequence-Recommender
cd DeepSequence-Recommender
pip install -r requirements.txt

 Training

python train.py --model transformer --epochs 20 --seed 42

Supported models:

python train.py --model gru
python train.py --model lstm
python train.py --model itemknn
python train.py --model popularity


Evaluation

python evaluate.py --model transformer --topk 10


Future Enhancements

Add GNN-based recommenders (SR-GNN, GCSAN)

Contrastive embedding pretraining

Metadata-aware hybrid models

FastAPI inference endpoint

Streamlit dashboard for visualization

Hyperparameter tuning (Optuna)

Distributed training (DeepSpeed, Horovod)



Corey Leath
GitHub: https://github.com/Trojan3877
LinkedIn: https://www.linkedin.com/in/corey-leath
Email: corey22blue@hotmail.com



📜 License

MIT License

