"""
NetRecommender-Capstone
Utility & Helper Functions (L5/L6 Production Quality)

Author: Corey Leath (Trojan3877)
"""

import os
import yaml
import time
import random
import numpy as np
import logging
import pandas as pd
import tensorflow as tf


# ---------------------------------------------------------------------------
# Load config.yaml
# ---------------------------------------------------------------------------
def load_config(path="config/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Ensure reproducibility (L6 requirement)
# ---------------------------------------------------------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------
def setup_logging(log_dir="artifacts/logs/"):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "pipeline.log")

    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logging.getLogger().addHandler(logging.StreamHandler())
    logging.info("Logging initialized.")
    return logging


# ---------------------------------------------------------------------------
# Timer Decorator (great for profiling)
# ---------------------------------------------------------------------------
def timing(func):
    def wrap(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"[TIMER] {func.__name__} took {end - start:.2f}s")
        return result
    return wrap


# ---------------------------------------------------------------------------
# Safe directory creation
# ---------------------------------------------------------------------------
def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


# ---------------------------------------------------------------------------
# Load dataset files with safe checks
# ---------------------------------------------------------------------------
def load_ratings(ratings_path: str) -> pd.DataFrame:
    if not os.path.exists(ratings_path):
        raise FileNotFoundError(f"Ratings file not found: {ratings_path}")

    df = pd.read_csv(ratings_path)
    return df


def load_items(items_path: str) -> pd.DataFrame:
    if not os.path.exists(items_path):
        raise FileNotFoundError(f"Items file not found: {items_path}")

    df = pd.read_csv(items_path)
    return df


# ---------------------------------------------------------------------------
# Train/Val/Test Split for recommenders
# Stratified by user to prevent leakage (L6 approach)
# ---------------------------------------------------------------------------
def user_stratified_split(df, test_size=0.2, val_size=0.1, seed=42):
    set_seed(seed)

    users = df["user_id"].unique()
    np.random.shuffle(users)

    n_users = len(users)
    n_test = int(n_users * test_size)
    n_val = int(n_users * val_size)

    test_users = users[:n_test]
    val_users = users[n_test:n_test + n_val]
    train_users = users[n_test + n_val:]

    train_df = df[df["user_id"].isin(train_users)]
    val_df = df[df["user_id"].isin(val_users)]
    test_df = df[df["user_id"].isin(test_users)]

    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# Save metrics to artifacts
# ---------------------------------------------------------------------------
def save_metrics(metrics: dict, output_dir="artifacts/metrics/"):
    ensure_dir(output_dir)
    path = os.path.join(output_dir, "metrics.json")

    import json
    with open(path, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"[INFO] Metrics saved to {path}")
