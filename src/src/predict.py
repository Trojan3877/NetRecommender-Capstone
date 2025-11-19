"""
NetRecommender-Capstone
Prediction & Recommendation Module (L6 Production Quality)

Author: Corey Leath (Trojan3877)

This module handles:
✔ Loading trained model
✔ Predicting user-item scores
✔ Generating Top-N recommendations
✔ Returning ranked item lists for API or batch inference
"""

import os
import numpy as np
import tensorflow as tf

from utils import load_config
from data_loader import load_recommender_dataset


# -------------------------------------------------------------------
# Load model
# -------------------------------------------------------------------
def load_trained_model(config_path="config/config.yaml"):
    config = load_config(config_path)
    model_path = os.path.join(config["paths"]["model_dir"], "best_model.keras")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"[ERROR] No model found at {model_path}. "
            f"Run train.py first to create the model."
        )

    print(f"[INFO] Loading model from: {model_path}")
    model = tf.keras.models.load_model(model_path)
    return model


# -------------------------------------------------------------------
# Predict score for a single user & item
# -------------------------------------------------------------------
def predict_single(model, user_id, item_id):
    preds = model.predict({"user": np.array([user_id]),
                           "item": np.array([item_id])}, verbose=0)
    return float(preds[0])


# -------------------------------------------------------------------
# Generate Top-N recommendations for a user
# -------------------------------------------------------------------
def recommend_top_k(model, user_idx, num_items, top_k=10):
    """
    Returns the top-K recommended items for a given user.
    """

    user_tensor = tf.constant([user_idx] * num_items)
    item_tensor = tf.constant(list(range(num_items)))

    # Predict all scores for (user, every item)
    preds = model.predict({"user": user_tensor,
                           "item": item_tensor}, verbose=0)
    preds = preds.reshape(-1)

    # Rank items
    ranked_items = np.argsort(preds)[::-1]

    return ranked_items[:top_k], preds[ranked_items[:top_k]]


# -------------------------------------------------------------------
# Pretty Print Recommendations
# -------------------------------------------------------------------
def print_recommendations(user_name, top_items, scores, item_reverse_map=None):
    print(f"\nTop Recommendations for User: {user_name}")
    print("-" * 50)

    for i, (item, score) in enumerate(zip(top_items, scores)):
        item_label = item_reverse_map[item] if item_reverse_map else item
        print(f"Rank {i+1}: {item_label} (score={score:.4f})")

    print("-" * 50)


# -------------------------------------------------------------------
# Full Inference Pipeline
# -------------------------------------------------------------------
def run_inference(user_name, config_path="config/config.yaml", top_k=10):

    # Load config + dataset
    config = load_config(config_path)
    data = load_recommender_dataset(config_path)

    model = load_trained_model(config_path)

    # Mapping user names → IDs
    user_map = data["user_map"]
    item_map = data["item_map"]
    item_reverse_map = {v: k for k, v in item_map.items()}

    # Validate user exists
    if user_name not in user_map:
        raise ValueError(
            f"[ERROR] User '{user_name}' not found in dataset. "
            "Check the raw dataset or ensure user exists in training data."
        )

    user_idx = user_map[user_name]
    num_items = data["num_items"]

    # Generate recommendations
    recommended_items, scores = recommend_top_k(
        model=model,
        user_idx=user_idx,
        num_items=num_items,
        top_k=top_k
    )

    # Display
    print_recommendations(
        user_name=user_name,
        top_items=recommended_items,
        scores=scores,
        item_reverse_map=item_reverse_map
    )

    return recommended_items, scores


# -------------------------------------------------------------------
# CLI Entry Point (Optional)
# -------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate recommendations for a user.")
    parser.add_argument("--user", type=str, required=True, help="Username for inference")
    parser.add_argument("--top_k", type=int, default=10, help="Number of recommendations")

    args = parser.parse_args()
    run_inference(user_name=args.user, top_k=args.top_k)
