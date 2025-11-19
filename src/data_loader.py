"""
NetRecommender-Capstone
Data Loader Module (L6 Production Quality)

Author: Corey Leath (Trojan3877)

This module provides:
✔ Loading raw interaction data
✔ Encoding users and items
✔ Train/val/test split
✔ Negative sampling for implicit feedback
✔ TensorFlow dataset generation (tf.data)
✔ Config-driven pipeline
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf

from utils import ensure_dir, load_config


# -------------------------------------------------------------
# Load the interactions CSV file
# -------------------------------------------------------------
def load_interactions_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"[ERROR] Interaction file not found at: {path}")

    df = pd.read_csv(path)

    required_cols = {"user_id", "item_id", "rating"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"[ERROR] CSV must contain columns: {required_cols}. Found: {df.columns}"
        )

    print(f"[INFO] Loaded interactions: {df.shape[0]} rows.")
    return df


# -------------------------------------------------------------
# Encode users & items to integer indices
# -------------------------------------------------------------
def encode_ids(df):
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()

    df["user_idx"] = user_encoder.fit_transform(df["user_id"])
    df["item_idx"] = item_encoder.fit_transform(df["item_id"])

    num_users = df["user_idx"].max() + 1
    num_items = df["item_idx"].max() + 1

    print(f"[INFO] Encoded {num_users} users and {num_items} items.")

    return df, num_users, num_items, user_encoder, item_encoder


# -------------------------------------------------------------
# Generate Negative Samples (implicit feedback)
# -------------------------------------------------------------
def generate_negative_samples(df, num_items, negative_ratio=4):
    """
    For every positive interaction, generate N negative samples.

    Example:
      If user liked item A → negative samples = items they did NOT interact with.
    """

    print("[INFO] Generating negative samples...")

    user_positive_items = (
        df.groupby("user_idx")["item_idx"].apply(set).to_dict()
    )

    users, items, labels = [], [], []

    for row in df.itertuples():
        # positive example
        users.append(row.user_idx)
        items.append(row.item_idx)
        labels.append(1)

        # generate negative examples
        for _ in range(negative_ratio):
            neg_item = np.random.randint(0, num_items)
            while neg_item in user_positive_items[row.user_idx]:
                neg_item = np.random.randint(0, num_items)

            users.append(row.user_idx)
            items.append(neg_item)
            labels.append(0)

    print("[INFO] Negative sampling complete.")

    return np.array(users), np.array(items), np.array(labels)


# -------------------------------------------------------------
# Build TensorFlow Dataset
# -------------------------------------------------------------
def build_tf_dataset(users, items, labels, batch_size=256, shuffle=True):

    ds = tf.data.Dataset.from_tensor_slices(
        (
            {
                "user": users.astype("int32"),
                "item": items.astype("int32"),
            },
            labels.astype("float32"),
        )
    )

    if shuffle:
        ds = ds.shuffle(buffer_size=len(users))

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# -------------------------------------------------------------
# Master pipeline function (called inside train.py)
# -------------------------------------------------------------
def load_dataset(config_path="config/config.yaml"):

    config = load_config(config_path)

    csv_path = config["paths"]["interactions"]
    batch_size = config["training"]["batch_size"]
    neg_ratio = config["training"]["negative_samples"]

    # Step 1 — Load CSV
    df = load_interactions_csv(csv_path)

    # Step 2 — Encode IDs
    df, num_users, num_items, user_encoder, item_encoder = encode_ids(df)

    # Step 3 — Train/Val/Test split
    train_df, test_df = train_test_split(df, test_size=0.10, random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.10, random_state=42)

    print(
        f"[INFO] Split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}"
    )

    # Step 4 — Create negative samples
    train_users, train_items, train_labels = generate_negative_samples(
        train_df, num_items, negative_ratio=neg_ratio
    )
    val_users, val_items, val_labels = generate_negative_samples(
        val_df, num_items, negative_ratio=neg_ratio
    )
    test_users, test_items, test_labels = generate_negative_samples(
        test_df, num_items, negative_ratio=neg_ratio
    )

    # Step 5 — Convert to TF datasets
    train_ds = build_tf_dataset(train_users, train_items, train_labels, batch_size)
    val_ds = build_tf_dataset(val_users, val_items, val_labels, batch_size, shuffle=False)
    test_ds = build_tf_dataset(test_users, test_items, test_labels, batch_size, shuffle=False)

    print("[INFO] TensorFlow datasets built successfully.")

    return {
        "train": train_ds,
        "val": val_ds,
        "test": test_ds,
        "num_users": num_users,
        "num_items": num_items,
        "user_encoder": user_encoder,
        "item_encoder": item_encoder,
    }
