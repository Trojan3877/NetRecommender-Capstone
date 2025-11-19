"""
NetRecommender-Capstone
Utility Functions (L6 Production Quality)

Author: Corey Leath (Trojan3877)

This module provides:
✔ Config loading with validation
✔ Safe directory creation
✔ Experiment logging
✔ Device (GPU/CPU) reporting
✔ Seed fixing for reproducibility
"""

import os
import yaml
import random
import numpy as np
import tensorflow as tf
from datetime import datetime


# -------------------------------------------------------------------
# Load YAML Configuration
# -------------------------------------------------------------------
def load_config(config_path="config/config.yaml"):
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"[ERROR] Config file not found at: {config_path}"
        )

    with open(config_path, "r") as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"[ERROR] Invalid YAML format: {e}")

    # Basic validation
    required_sections = ["training", "paths", "model"]
    for section in required_sections:
        if section not in config:
            raise ValueError(f"[ERROR] Missing '{section}' section in config.yaml")

    return config


# -------------------------------------------------------------------
# Make directory safely
# -------------------------------------------------------------------
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        print(f"[INFO] Created directory: {path}")


# -------------------------------------------------------------------
# Fix Random Seeds
# -------------------------------------------------------------------
def set_global_seed(seed=42):
    print(f"[INFO] Setting global seed: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# -------------------------------------------------------------------
# GPU/CPU Device Report
# -------------------------------------------------------------------
def print_device_info():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"[INFO] GPU is available: {gpus}")
    else:
        print("[INFO] Running on CPU (no GPU detected)")


# -------------------------------------------------------------------
# Time helper
# -------------------------------------------------------------------
def timestamp():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


# -------------------------------------------------------------------
# Training log header
# -------------------------------------------------------------------
def print_training_header(config):
    print("\n" + "=" * 60)
    print("        NET RECOMMENDER — TRAINING SESSION STARTED")
    print("=" * 60)
    print(f"Timestamp: {timestamp()}")
    print(f"Epochs: {config['training']['epochs']}")
    print(f"Batch Size: {config['training']['batch_size']}")
    print(f"Learning Rate: {config['training']['learning_rate']}")
    print("=" * 60 + "\n")


# -------------------------------------------------------------------
# Summarize model after training
# -------------------------------------------------------------------
def summarize_training_history(history):
    print("\n" + "=" * 60)
    print("               TRAINING SUMMARY")
    print("=" * 60)

    for key in history.history:
        last_val = history.history[key][-1]
        print(f"{key}: {last_val:.4f}")

    print("=" * 60 + "\n")


# -------------------------------------------------------------------
# Save training metrics to disk
# -------------------------------------------------------------------
def save_metrics(history, path="results/metrics.txt"):
    ensure_dir(os.path.dirname(path))

    with open(path, "w") as f:
        f.write("Training Metrics\n")
        f.write("=" * 40 + "\n")
        for key in history.history:
            values = [str(round(v, 5)) for v in history.history[key]]
            f.write(f"{key}: {values}\n")

    print(f"[INFO] Saved metrics → {path}")
