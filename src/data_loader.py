"""
NetRecommender-Capstone
Data Loader & Preprocessing Pipeline (L5/L6 Production Quality)

Author: Corey Leath (Trojan3877)

Handles:
✔ Load and validate ratings data
✔ User-item indexing (ID encoding)
✔ User-stratified train/val/test split
✔ Negative sampling for implicit models
✔ Conversion to TensorFlow Datasets
✔ Fully config-driven
"""

import os
import numpy as np
import pandas as pd
from utils import (
    load_config,
    load_ratings,
    load_items,
    user_stratified_split,
    ensure_dir,
    set_seed,
)


# ---------------------------------------------------------------------------
# Encode user and item IDs (Netflix-style indexing)
# ---------------------------------------------------------------------------
def encode_ids(df):
    user_ids = sorted(df["user_id"].unique())
    item_ids = sorted(df["item_id"].unique())

    user_map = {u: idx for idx, u in enumerate(user_ids)}
    item_map = {i: idx for idx, i in enumerate(item_ids)}

    df["user_idx"] = df["user_id"].map(user_map)
    df["item_idx"] = df["item_id"].map(item_map)

    return df, user_map, item_map


# ---------------------------------------------------------------------------
# Negative sampling for implicit recommenders (L6 standard)
# ---------------------------------------------------------------------------
def generate_negative_samples(df, num_items, neg_ratio=4):
    """
    For every positive user-item pair, generate K negative samples.
    Ensures training stability and performance.
    """
    negatives = []
    user_positive_items = df.groupby("user_idx")["item_idx"].apply(set).to_dict()

    for _, row in df.iterrows():
        user = row["user_idx"]

        for _ in range(neg_ratio):
            neg_item = np.random.randint(0, num_items)

            # Ensure the negative sample is truly negative
            while neg_item in user_positive_items[user]:
                neg_item = np.random.randint(0, num_items)

            negatives.append([user, neg_item, 0])  # label = 0

    positives = df[["user_idx", "item_idx"]].copy()
    positives["label"] = 1

    neg_df = pd.DataFrame(negatives, columns=["user_idx", "item_idx", "label"])
    pos_df = positives[["user_idx", "item_idx", "label"]]

    return pd.concat([pos_df, neg_df], axis=0).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Build TensorFlow Datasets
# ---------------------------------------------------------------------------
def build_tf_dataset(df, batch_size=128, shuffle=True):
    import tensorflow as tf

    dataset = tf.data.Dataset.from_tensor_slices(
        (
            {
                "user": df["user_idx"].values,
                "item": df["item_idx"].values,
            },
            df["label"].values,
        )
    )

    if shuffle:
        dataset = dataset.shuffle(10_000)

    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


# ---------------------------------------------------------------------------
# Full data pipeline
# ---------------------------------------------------------------------------
def load_recommender_dataset(config_path="config/config.yaml"):

    config = load_config(config_path)
    set_seed(config["dataset"]["seed"])

    # Paths
    ratings_path = os.path.join(config["dataset"]["path"], config["dataset"]["ratings_file"])
    items_path = os.path.join(config["dataset"]["path"], config["dataset"]["items_file"])

    # Load files
    ratings_df = load_ratings(ratings_path)
    items_df = load_items(items_path)

    # Remove low-activity users
    min_ratings = config["dataset"]["min_ratings_per_user"]
    ratings_df = ratings_df.groupby("user_id").filter(lambda x: len(x) >= min_ratings)

    # Encode integer IDs
    ratings_df, user_map, item_map = encode_ids(ratings_df)

    num_users = len(user_map)
    num_items = len(item_map)

    # Split
    train_df, val_df, test_df = user_stratified_split(
        ratings_df,
        test_size=config["dataset"]["test_size"],
        val_size=config["dataset"]["val_size"],
        seed=config["dataset"]["seed"],
    )

    # Generate negative samples for each split
    train_df = generate_negative_samples(train_df, num_items)
    val_df = generate_negative_samples(val_df, num_items, neg_ratio=2)
    test_df = generate_negative_samples(test_df, num_items, neg_ratio=2)

    # Convert to TF datasets
    train_ds = build_tf_dataset(
        train_df,
        batch_size=config["training"]["batch_size"],
        shuffle=config["training"]["shuffle"],
    )

    val_ds = build_tf_dataset(
        val_df,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
    )

    test_ds = build_tf_dataset(
        test_df,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
    )

    return {
        "train": train_ds,
        "val": val_ds,
        "test": test_ds,
        "num_users": num_users,
        "num_items": num_items,
        "user_map": user_map,
        "item_map": item_map,
    }
