# NetRecommender-Capstone

<p align="left"> <img src="https://img.shields.io/badge/Framework-TensorFlow-orange?style=flat-square" /> <img src="https://img.shields.io/badge/Python-3.10-blue?style=flat-square" /> <img src="https://img.shields.io/badge/Deep%20Learning-NCF%20%7C%20NeuMF-purple?style=flat-square" /> <img src="https://img.shields.io/badge/MLOps-Docker%20%7C%20GPU%20Ready-green?style=flat-square" /> <img src="https://img.shields.io/badge/RecSys-Precision%40K%20%7C%20NDCG%20%7C%20Recall@K-yellow?style=flat-square" /> <img src="https://img.shields.io/badge/Status-Production%20Ready-brightgreen?style=flat-square" /> <img src="https://img.shields.io/badge/Level-L5%2FL6%20FAANG%20Quality-red?style=flat-square" /> </p>


🎯 Overview

NetRecommender-Capstone is a full end-to-end, FAANG-level deep learning recommendation engine.
It implements Neural Collaborative Filtering (NCF) — the same architecture powering:

Netflix Personalized Ranking

TikTok For-You Feed

YouTube Deep Recommendations

Spotify Homefeed

Amazon Personalize

This repository is designed to demonstrate L6 ML Engineering, MLOps, and RecSys skills.

![Architecture](assets/NetRecommender_Architecture.png)

🧠 Key Features
✔ Neural Collaborative Filtering (NCF)

Trainable user embeddings

Trainable item embeddings

Configurable MLP layers

Dropout & L2 regularization

Sigmoid prediction for implicit data

✔ Complete Training Pipeline

Negative Sampling (4:1 ratio)

GPU-accelerated training

TF tf.data pipeline for batching/shuffling

Checkpoint saving

Training logs + metrics export

✔ FAANG-Level Evaluation Metrics

Precision@K

Recall@K

NDCG@K

Hit Rate@K

✔ Production-Ready MLOps

GPU-enabled Dockerfile

Version-pinned requirements.txt

Clean modular folder structure

Fully configurable config.yaml


## Project layout
## Quick Start (Docker)

```bash
# 1) Start all services
docker compose up -d --build

# 2) Open the UIs
# Airflow:   http://localhost:8080  (user: admin / pw: admin)
# MLflow:    http://localhost:5000
# API docs:  http://localhost:8000/docs
# Dashboard: http://localhost:8501


## 📦 Demo Files

- 📁 [`data/`](./data/README.md): Dataset source and usage
- 🧪 [`api/examples/example_request.json`](./api/examples/example_request.json): Sample input
- 🧾 [`api/examples/example_response.json`](./api/examples/example_response.json): Sample output
- 📓 [`notebooks/demo_usage.ipynb`](./notebooks/demo_usage.ipynb): Demo workflow and visualization

---

## 📘 Extended Project Overview

**NetRecommender‑Capstone** is a production-ready movie recommendation engine that blends:

- **Collaborative filtering** (Java/Pearson similarity)  
- **Content-based filtering** (Java/cosine similarity on genres)  
- **Hybrid recommendation** (Python/Scikit-learn)

...all exposed via a **FastAPI** and interactive **Streamlit dashboard**, deployed to **Kubernetes** via **Helm**, automated with **Terraform**, **Ansible**, and **GitHub Actions**. Ideal for showcasing end-to-end ML + DevOps practices at scale.

---

## 📂 Project Structure
NetRecommender-Capstone/
│
├── config/
│   └── config.yaml
│
├── data/
│   └── interactions.csv
│
├── src/
│   ├── data_loader.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── utils.py
│
├── checkpoints/
├── results/
├── logs/
│
├── Dockerfile
├── requirements.txt
└── README.md

         ┌───────────────────┐
         │   user_id (int)   │
         └─────────┬─────────┘
                   ▼
        ┌──────────────────────┐
        │  User Embedding (k)  │
        └──────────────────────┘

         ┌───────────────────┐
         │   item_id (int)   │
         └─────────┬─────────┘
                   ▼
        ┌──────────────────────┐
        │ Item Embedding (k)   │
        └──────────────────────┘

                Concatenate
                   │
                   ▼
         ┌────────────────────────────┐
         │   MLP Layers (128→64→32)   │
         └────────────────────────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │ Sigmoid Output Layer │
        └──────────────────────┘
                   │
                   ▼
         "Recommended or Not"


| Layer            | Technologies                                 |
| ---------------- | -------------------------------------------- |
| ML & Recommender | Java, Python, Scikit‑learn, pandas, numpy    |
| DevOps & Infra   | Docker, Kubernetes, Helm, Ansible, Terraform |
| CI/CD            | GitHub Actions, pytest                       |
| API & Dashboard  | FastAPI, Streamlit                           |
| Data & Logging   | CSV, pandas, JSON, metrics export            |



![image](https://github.com/user-attachments/assets/cee7ca96-aa5d-403d-bb83-ef65a352ef3b)
