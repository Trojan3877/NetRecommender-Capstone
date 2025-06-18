# 📊 Netflix-Style Recommendation Engine – System Architecture

## 🧠 Core Design

This project implements a **Hybrid Recommendation System** using both:
- **Collaborative Filtering (KNN):** Leverages user-user similarity.
- **Content-Based Filtering (TF-IDF on genres):** Matches user preferences with item attributes.

The hybrid model blends both methods with an adjustable weight parameter (α).

---

## 📂 Modular Structure


---

## ☁ Deployment Stack

| Layer         | Tool                         |
|--------------|------------------------------|
| Infrastructure | AWS EKS (via Terraform)     |
| Orchestration | Kubernetes + Helm            |
| Automation    | Ansible + GitHub Actions     |
| ML Stack      | Python, Pandas, scikit-learn |
| Data Storage  | Snowflake (optional)         |

---

## 📈 CI/CD Pipeline

1. Developer pushes to `main`
2. GitHub Actions triggers `train_pipeline.py`
3. Models validate + log recommendations
4. Optional: Deploys to Kubernetes via Helm & Terraform

---

## 📊 Performance Goals

- 🧪 Accuracy testing in progress using RMSE and Precision@K
- ⚙ Scalable to millions of users/movies with EKS
- 📦 Containerized for reproducible deployments

---

## 🔒 Future Improvements

- Add real-time recommendation API with FastAPI
- Integrate Snowflake queries for dynamic data ingestion
- Use SageMaker for hyperparameter tuning

## ☁ Deployment Stack

| Layer         | Tool                         |
|--------------|------------------------------|
| Infrastructure | AWS EKS (via Terraform)     |
| Orchestration | Kubernetes + Helm            |
| Automation    | Ansible + GitHub Actions     |
| ML Stack      | Python, Pandas, scikit-learn |
| Data Storage  | Snowflake (optional)         |

---

## 📈 CI/CD Pipeline

1. Developer pushes to `main`
2. GitHub Actions triggers `train_pipeline.py`
3. Models validate + log recommendations
4. Optional: Deploys to Kubernetes via Helm & Terraform

---

## 📊 Performance Goals

- 🧪 Accuracy testing in progress using RMSE and Precision@K
- ⚙ Scalable to millions of users/movies with EKS
- 📦 Containerized for reproducible deployments

---

## 🔒 Future Improvements

- Add real-time recommendation API with FastAPI
- Integrate Snowflake queries for dynamic data ingestion
- Use SageMaker for hyperparameter tuning
