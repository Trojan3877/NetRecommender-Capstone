"""
NetRecommender-Capstone
Training Pipeline (L5/L6 Production Quality)

Author: Corey Leath (Trojan3877)

Handles:
✔ Loading dataset (from data_loader.py)
✔ Building NCF model (from model.py)
✔ Early stopping + checkpointing
✔ Training history export
✔ Logging + reproducibility
✔ Artifact directory management
"""

import os
import json
import tensorflow as tf

from utils import load_config, ensure_dir, setup_logging, set_seed
from data_loader import load_recommender_dataset
from model import build_ncf_model


# -----------------------------------------------------------
# Main training function
# -----------------------------------------------------------
def train_model(config_path="config/config.yaml"):

    # Load config
    config = load_config(config_path)
    set_seed(config["dataset"]["seed"])

    # Setup logging
    logger = setup_logging(log_dir=config["paths"]["logs_dir"])
    logger.info("Starting training process...")

    # Load dataset
    data = load_recommender_dataset(config_path)
    train_ds, val_ds, test_ds = data["train"], data["val"], data["test"]
    num_users, num_items = data["num_users"], data["num_items"]

    # Build model
    model = build_ncf_model(num_users, num_items, config_path=config_path)
    model.summary(print_fn=lambda x: logger.info(x))

    # Create artifact directories
    model_dir = config["paths"]["model_dir"]
    metrics_dir = config["paths"]["metrics_dir"]

    ensure_dir(model_dir)
    ensure_dir(metrics_dir)

    # -------------------------------------------------------
    # Callbacks (L6 production-grade)
    # -------------------------------------------------------
    checkpoint_path = os.path.join(model_dir, "best_model.keras")

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_best_only=True,
        monitor="val_loss",
        mode="min",
        verbose=1,
    )

    early_stop_cb = tf.keras.callbacks.EarlyStopping(
        patience=config["training"]["early_stopping_patience"],
        monitor="val_loss",
        mode="min",
        restore_best_weights=True,
        verbose=1,
    )

    # -------------------------------------------------------
    # Train the model
    # -------------------------------------------------------
    logger.info("Training model...")

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config["training"]["epochs"],
        callbacks=[checkpoint_cb, early_stop_cb],
        verbose=1,
    )

    # Save training history
    hist_path = os.path.join(model_dir, "training_history.json")
    with open(hist_path, "w") as f:
        json.dump(history.history, f, indent=4)

    logger.info(f"Training history saved to {hist_path}")

    # -------------------------------------------------------
    # Evaluate on test set
    # -------------------------------------------------------
    logger.info("Evaluating best model on test set...")
    best_model = tf.keras.models.load_model(checkpoint_path)

    test_loss, test_acc = best_model.evaluate(test_ds, verbose=1)

    metrics = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
    }

    # Save metrics
    metrics_path = os.path.join(metrics_dir, "evaluation_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    logger.info(f"Metrics saved to {metrics_path}")

    logger.info("Training complete.")
    return metrics
