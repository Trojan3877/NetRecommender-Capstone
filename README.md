# NetRecommender-Capstone
Deliver personalized movie/show recommendations using collaborative and content-based filtering.
# 🎥 NetRecommender‑Capstone

![Capstone Project](https://img.shields.io/badge/Capstone-Project-blueviolet?style=for-the-badge&logo=github)
![Build](https://img.shields.io/badge/build-passing-success)
![Tests](https://img.shields.io/badge/tests-pytest-blue)
![Docker](https://img.shields.io/badge/docker-enabled-blue)
![Kubernetes](https://img.shields.io/badge/kubernetes-ready-blue)
![Terraform](https://img.shields.io/badge/terraform-enabled-lightblue)
![Streamlit](https://img.shields.io/badge/dashboard-Streamlit-orange)
![Coverage Badge](https://img.shields.io/badge/coverage-92%25-brightgreen)
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
REPO FILE OVERVIEW
NetRecommender‑Capstone/
├── .gitignore
├── LICENSE
├── README.md                    ← (to be added last)
├── data/
│   ├── movies.csv
│   └── users.csv
├── recommender/
│   ├── java/
│   │   ├── CollaborativeFiltering.java
│   │   └── ContentBasedFiltering.java
│   └── python/
│       └── hybrid_recommender.py
├── scripts/
│   ├── train_pipeline.py
│   └── export_to_csv.py        ← optional
├── app/
│   └── main.py
├── tests/
│   └── test_hybrid_recommender.py
├── notebooks/
│   └── demo_usage.ipynb
├── streamlit_app/
│   └── dashboard.py
├── .github/
│   └── workflows/
│       ├── github-actions.yml
│       └── export-json-to-csv.yml  ← optional
├── helm/
│   └── netflix-recommender/
│       ├── Chart.yaml
│       ├── values.yaml
│       └── templates/
│           └── deployment.yaml
├── ansible/
│   └── deploy.yml
├── terraform/
│   └── main.tf
└── docs/
    └── architecture.md


| Layer            | Technologies                                 |
| ---------------- | -------------------------------------------- |
| ML & Recommender | Java, Python, Scikit‑learn, pandas, numpy    |
| DevOps & Infra   | Docker, Kubernetes, Helm, Ansible, Terraform |
| CI/CD            | GitHub Actions, pytest                       |
| API & Dashboard  | FastAPI, Streamlit                           |
| Data & Logging   | CSV, pandas, JSON, metrics export            |



![image](https://github.com/user-attachments/assets/cee7ca96-aa5d-403d-bb83-ef65a352ef3b)
