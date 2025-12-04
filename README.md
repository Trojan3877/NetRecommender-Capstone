<!-- Banner -->
<p align="center">
  <img src="https://raw.githubusercontent.com/Trojan3877/assets/main/deepsequence_banner_dark.png" width="100%" alt="DeepSequence-Recommender Banner"/>
</p>

<h1 align="center">DeepSequence-Recommender</h1>
<p align="center">
  A production-ready recommendation system integrating Deep Learning, Neural Collaborative Filtering, and API deployment. Built to L5/L6 Machine Learning Engineering standards.
</p>

---

# 🚀 Overview

**DeepSequence-Recommender** is an end-to-end machine learning system designed to model, train, evaluate, and serve recommendation models using industry-standard methods.

This project mirrors the engineering patterns used at **Netflix, TikTok, Amazon, Spotify, and YouTube**, and includes:

- Matrix Factorization (MF)
- Neural Collaborative Filtering (NCF)
- FastAPI inference service
- Dockerized deployment pipeline
- Ranking metrics (Precision@K, Recall@K, MAP@K, NDCG@K, HitRate)
- Clean L5/L6 production engineering structure

Transformer sequential models (**SASRec-style**) can be added easily with the provided modular architecture.

---



<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue"/>
  <img src="https://img.shields.io/badge/PyTorch-2.2-red"/>
  <img src="https://img.shields.io/badge/FastAPI-Production-green"/>
  <img src="https://img.shields.io/badge/Docker-Ready-blue"/>
  <img src="https://img.shields.io/badge/ML%20Engineering-L5%2FL6-purple"/>
  <img src="https://img.shields.io/badge/Transformers-Ready-orange"/>
  <img src="https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-yellow"/>
</p>

---

# 🏗 Project Architecture (Light Theme)

                     +-----------------------+
                     |     Ratings CSV       |
                     +-----------+-----------+
                                 |
                                 v
                     +-----------------------+
                     |   Data Preprocessing  |
                     |  - Encode IDs         |
                     |  - Normalize Ratings  |
                     +-----------+-----------+
                                 |
                                 v
     +------------------------------------------------------------+
     |                        Training Pipeline                    |
     |-------------------------------------------------------------|
     |  Config-driven training (MF or NCF)                         |
     |  PyTorch model training                                    |
     |  GPU/CPU auto-detection                                    |
     |  Checkpoint saving                                          |
     +-------------------------------------------------------------+
                                 |
                                 v
                     +-----------------------+
                     |      Checkpoints      |
                     +-----------+-----------+
                                 |
                                 v
              +-------------------------------------+
              |        Inference Pipeline           |
              |  - Load checkpoint                  |
              |  - Recommend top-K items            |
              |  - Device-aware prediction          |
              +------------------+------------------+
                                 |
                                 v
                      +-----------------------+
                      |      FastAPI API      |
                      |   /recommend/{id}     |
                      +-----------------------+

---
![Uploading image.png…]()

# 🔄 System Flow (Light Theme)


---

# 🧠 Models Included

### ✔ **Matrix Factorization (MF)**
Simple, interpretable baseline using:

- User embeddings  
- Item embeddings  
- Bias terms  
- Dot product  

### ✔ **Neural Collaborative Filtering (NCF)**
Deep model using:

- Embeddings  
- MLP tower  
- LayerNorm  
- Dropout  
- Sigmoid output  

### ✔ **Transformer Ready (SASRec / GPT-like)**
Your repo is structured to support:

- Multi-head attention  
- Positional encoding  
- Sequential modeling  
- Next-item prediction  

(Generated upon request.)

---

# 📦 Tech Stack

- **Python 3.10**
- **PyTorch**
- **FastAPI + Uvicorn**
- **Docker**
- **YAML configs**
- **Pandas / NumPy**
- **Evaluation metrics for ranking**
- **Neural Network architectures**

---

# 🏋️ Training Pipeline

```bash
python main.py
uvicorn src.api.fastapi_app:app --reload
GET /recommend/{user_id}?top_k=10
{
  "user_id": "123",
  "top_k": 10,
  "recommendations": [
    {"item_id": "A1", "score": 0.91},
    {"item_id": "B4", "score": 0.87}
  ]
}
docker build -t deepsequence-recommender .
docker run -p 8000:8000 deepsequence-recommender

Academic Pathway 

This project is part of My journey toward becoming a Machine Learning Engineer and earning a Master's in AI Engineering followed by a PhD in Artificial Intelligence.

By building end-to-end production-grade systems like DeepSequence-Recommender, Corey demonstrates:

Mastery of software engineering

Applied machine learning

Cloud & containerization

FastAI/FastAPI deployment

Big Tech–level project quality
