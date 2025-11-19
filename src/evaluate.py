"""
NetRecommender-Capstone
Evaluation Metrics for Recommender Systems (L6 Production Quality)

Author: Corey Leath (Trojan3877)

Metrics Implemented:
✔ Precision@K
✔ Recall@K
✔ NDCG@K
✔ HitRate@K

These are industry-standard metrics used by Netflix, YouTube, Spotify,
Amazon Personalize, TikTok, and other ranking-based recommender systems.
"""

import numpy as np
from tqdm import tqdm


# ---------------------------------------------------------------------
# Compute DCG (Discounted Cumulative Gain)
# ---------------------------------------------------------------------
def dcg_at_k(relevance_list, k):
    relevance_list = np.asfarray(relevance_list)[:k]
    if relevance_list.size:
        return np.sum(relevance_list / np.log2(np.arange(2, relevance_list.size + 2)))
    return 0.0


# ---------------------------------------------------------------------
# Compute NDCG@K
# ---------------------------------------------------------------------
def ndcg_at_k(relevance_list, k):
    dcg = dcg_at_k(relevance_list, k)
    ideal_dcg = dcg_at_k(sorted(relevance_list, reverse=True), k)

    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0


# ---------------------------------------------------------------------
# Precision@K
# ---------------------------------------------------------------------
def precision_at_k(relevance_list, k):
    relevance_list = np.asarray(relevance_list)[:k]
    return np.mean(relevance_list)


# ---------------------------------------------------------------------
# Recall@K
# ---------------------------------------------------------------------
def recall_at_k(relevance_list, k, total_positives):
    relevance_list = np.asarray(relevance_list)[:k]
    if total_positives == 0:
        return 0
    return np.sum(relevance_list) / total_positives


# ---------------------------------------------------------------------
# Hit Rate@K (Did we recommend at least one correct item?)
# ---------------------------------------------------------------------
def hit_rate_at_k(relevance_list, k):
    relevance_list = np.asarray(relevance_list)[:k]
    return 1.0 if np.sum(relevance_list) > 0 else 0.0


# ---------------------------------------------------------------------
# Rank items for a single user
# ---------------------------------------------------------------------
def rank_user(model, user_id, all_items):
    """
    Predict scores for all items for a given user.
    Returns sorted item indices (highest-score first).
    """
    user_arr = np.full(len(all_items), user_id, dtype="int32")
    item_arr = np.array(all_items, dtype="int32")

    preds = model.predict({"user": user_arr, "item": item_arr}, verbose=0)
    scores = preds.reshape(-1)

    return np.argsort(-scores)  # sort descending


# ---------------------------------------------------------------------
# Evaluate the model across a user test set
# ---------------------------------------------------------------------
def evaluate_model(model, test_df, num_items, k=10):
    """
    test_df must contain columns:
        user_idx, item_idx, rating (1 for positive interactions)

    This function computes:
    ✔ Precision@K
    ✔ Recall@K
    ✔ NDCG@K
    ✔ HitRate@K
    """

    print(f"[INFO] Evaluating model using Top-{k} metrics...")

    unique_users = test_df["user_idx"].unique()
    all_items = np.arange(num_items)

    precision_scores = []
    recall_scores = []
    ndcg_scores = []
    hr_scores = []

    for user in tqdm(unique_users, desc="Evaluating users"):

        # Positive interactions for this user
        user_pos_items = (
            test_df[test_df["user_idx"] == user]["item_idx"].values
        )

        if len(user_pos_items) == 0:
            continue

        # Get ranked list of items from the model
        ranked_items = rank_user(model, user, all_items)

        # Build relevance vector (1 = relevant, 0 = not relevant)
        relevance = [1 if item in user_pos_items else 0 for item in ranked_items]

        precision_scores.append(precision_at_k(relevance, k))
        recall_scores.append(recall_at_k(relevance, k, len(user_pos_items)))
        ndcg_scores.append(ndcg_at_k(relevance, k))
        hr_scores.append(hit_rate_at_k(relevance, k))

    results = {
        "precision@k": float(np.mean(precision_scores)),
        "recall@k": float(np.mean(recall_scores)),
        "ndcg@k": float(np.mean(ndcg_scores)),
        "hit_rate@k": float(np.mean(hr_scores)),
    }

    print("\n[INFO] Evaluation Results")
    print("===============================")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")
    print("===============================\n")

    return results
