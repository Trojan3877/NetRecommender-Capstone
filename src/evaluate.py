"""
NetRecommender-Capstone
Evaluation Module (L5/L6 Production Quality)

Author: Corey Leath (Trojan3877)

Evaluates:
✔ RMSE
✔ Precision@K
✔ Recall@K
✔ NDCG@K
✔ Ranking metrics for recommender systems
"""

import os
import json
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from utils import load_config, ensure_dir, save_metrics
from data_loader import load_recommender_dataset


# ---------------------------------------------------------------
# RMSE (Explicit or proxy metric)
# ---------------------------------------------------------------
def compute_rmse(model, test_ds):
    mse = tf.keras.metrics.MeanSquaredError()
    for batch_x, batch_y in test_ds:
        preds = model.predict(batch_x)
        mse.update_state(batch_y, preds)
    return np.sqrt(mse.result().numpy())


# ---------------------------------------------------------------
# Ranking Metrics
# ---------------------------------------------------------------
def precision_at_k(recommended, relevant, k):
    recommended_k = recommended[:k]
    return len(set(recommended_k) & set(relevant)) / k


def recall_at_k(recommended, relevant, k):
    recommended_k = recommended[:k]
    if len(relevant) == 0:
        return 0.0
    return len(set(recommended_k) & set(relevant)) / len(relevant)


def ndcg_at_k(recommended, relevant, k):
    recommended_k = recommended[:k]

    dcg = 0.0
    for idx, item in enumerate(recommended_k):
        if item in relevant:
            dcg += 1 / np.log2(idx + 2)

    ideal_dcg = 0.0
    for idx in range(min(len(relevant), k)):
        ideal_dcg += 1 / np.log2(idx + 2)

    if ideal_dcg == 0:
        return 0.0
    return dcg / ideal_dcg


# ---------------------------------------------------------------
# Generate Top-N Recommendations
# ---------------------------------------------------------------
def recommend_for_user(model, user_idx, num_items, top_k):
    user_tensor = tf.constant([user_idx] * num_items)
    item_tensor = tf.constant(list(range(num_items)))

    preds = model.predict({"user": user_tensor, "item": item_tensor}, verbose=0)
    preds = preds.reshape(-1)

    ranked_items = np.argsort(preds)[::-1]
    return ranked_items[:top_k]


# ---------------------------------------------------------------
# Full evaluation pipeline
# ---------------------------------------------------------------
def evaluate_model(config_path="config/config.yaml"):

    config = load_config(config_path)

    # Load dataset (includes ID maps)
    data = load_recommender_dataset(config_path)
    test_ds = data["test"]
    num_users = data["num_users"]
    num_items = data["num_items"]
    user_map = data["user_map"]
    item_map = data["item_map"]

    # Load best model
    best_model_path = os.path.join(config["paths"]["model_dir"], "best_model.keras")
    model = tf.keras.models.load_model(best_model_path)

    metrics = {}

    # -----------------------------------------------------------
    # RMSE
    # -----------------------------------------------------------
    if config["evaluation"]["compute_rmse"]:
        rmse = compute_rmse(model, test_ds)
        metrics["rmse"] = float(rmse)
        print(f"[RMSE] {rmse:.4f}")

    # -----------------------------------------------------------
    # Ranking Metrics (Precision@K, Recall@K, NDCG@K)
    # -----------------------------------------------------------
    if config["evaluation"]["compute_top_k"]:

        ks = config["evaluation"]["k_values"]
        precision_scores = {f"precision@{k}": [] for k in ks}
        recall_scores = {f"recall@{k}": [] for k in ks}
        ndcg_scores = {f"ndcg@{k}": [] for k in ks}

        print("[INFO] Evaluating ranking metrics...")

        for user in tqdm(range(num_users)):

            # Ground truth (positive interactions)
            relevant = set()
            for batch_x, batch_y in test_ds:
                mask = (batch_x["user"].numpy() == user) & (batch_y.numpy() == 1)
                relevant.update(batch_x["item"][mask].numpy())

            # Skip users without positive samples
            if len(relevant) == 0:
                continue

            # Generate top-N recommendations
            recommended = recommend_for_user(
                model=model,
                user_idx=user,
                num_items=num_items,
                top_k=max(ks)
            )

            # Compute metrics
            for k in ks:
                precision_scores[f"precision@{k}"].append(
                    precision_at_k(recommended, relevant, k)
                )
                recall_scores[f"recall@{k}"].append(
                    recall_at_k(recommended, relevant, k)
                )
                ndcg_scores[f"ndcg@{k}"].append(
                    ndcg_at_k(recommended, relevant, k)
                )

        # Aggregate mean results
        for k in ks:
            metrics[f"precision@{k}"] = float(np.mean(precision_scores[f"precision@{k}"]))
            metrics[f"recall@{k}"] = float(np.mean(recall_scores[f"recall@{k}"]))
            metrics[f"ndcg@{k}"] = float(np.mean(ndcg_scores[f"ndcg@{k}"]))

            print(f"Precision@{k}: {metrics[f'precision@{k}']:.4f}")
            print(f"Recall@{k}: {metrics[f'recall@{k}']:.4f}")
            print(f"NDCG@{k}: {metrics[f'ndcg@{k}']:.4f}")

    # -----------------------------------------------------------
    # Save metrics
    # -----------------------------------------------------------
    ensure_dir(config["paths"]["metrics_dir"])
    save_metrics(metrics, config["paths"]["metrics_dir"])

    print("\n[INFO] Evaluation complete.")
    return metrics
