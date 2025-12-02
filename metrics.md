# 📊 NetRecommender — Ranking Metrics Report

This document summarizes the ranking performance of the **NetRecommender** system using industry-standard evaluation metrics for recommender systems. These metrics are the same ones used by Netflix, Amazon, Spotify, TikTok, and YouTube to measure recommendation quality.

---

## 🧪 Evaluation Protocol

We use the following evaluation strategy:

- **Leave-One-Out** style ranking evaluation  
- For each user:
  - Hold out one or more real interactions as **true_items**
  - Rank all items using the trained model
  - Compute top-K performance metrics
- Metrics averaged across all users  
- Top-K = **10** unless otherwise specified  
- Dataset: `ratings.csv` (user-item interactions)

---

# 📈 Metrics Definitions

### **Precision@K**
How many of the top-K recommended items were actually relevant.

Formula:  
`Precision@K = (# relevant recommended items) / K`

---

### **Recall@K**
Of all relevant items, how many appear in the top-K list.

Formula:  
`Recall@K = (# relevant in top-K) / (# total relevant items)`

---

### **NDCG@K** (Normalized Discounted Cumulative Gain)
Rewards correct ranking *and* ordering.  
Higher value = better ranking alignment.

---

### **MAP@K** (Mean Average Precision)
Measures ranking quality with emphasis on **early** correct predictions.

---

### **HitRate@K**
1 if ANY relevant item appears in top-K, else 0.

---

# 🧾 **Final Evaluation Scores (Example Output)**

These scores come from `main.py` using the `RankingEvaluator`.

```text
precision@10: 0.1870
recall@10: 0.1625
ndcg@10: 0.2054
map@10: 0.1487
hitrate@10: 0.7910
