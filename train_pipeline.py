# src/pipelines/train_pipeline.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from src.models.baseline_cf import MatrixFactorization
from src.models.neural_recommender import NeuralRecommender
from src.data.preprocess import DataPreprocessor

import yaml
import os


# ----------------------------------------------------
# Custom PyTorch Dataset
# ----------------------------------------------------
class RatingsDataset(Dataset):
    """
    Simple dataset for user-item-rating triples.
    """

    def __init__(self, df):
        self.users = torch.tensor(df["user_idx"].values, dtype=torch.long)
        self.items = torch.tensor(df["item_idx"].values, dtype=torch.long)
        self.labels = torch.tensor(df["rating_normalized"].values, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.users[index], self.items[index], self.labels[index]


# ----------------------------------------------------
# Training Pipeline
# ----------------------------------------------------
class TrainingPipeline:
    """
    End-to-end training pipeline for both:
    - Baseline CF (Matrix Factorization)
    - Neural CF (Deep Learning Recommender)

    Controlled by configs/train_config.yaml
    """

    def __init__(self, config_path="configs/train_config.yaml"):
        self.config = self.load_config(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_config(self, path):
        with open(path, "r") as f:
            return yaml.safe_load(f)

    # ------------------------------------------------
    # Build selected model
    # ------------------------------------------------
    def build_model(self, num_users, num_items):
        model_type = self.config["model_type"]

        if model_type == "matrix_factorization":
            model = MatrixFactorization(
                num_users=num_users,
                num_items=num_items,
                embedding_dim=self.config["embedding_dim"],
                use_sigmoid=True
            )

        elif model_type == "neural_cf":
            model = NeuralRecommender(
                num_users=num_users,
                num_items=num_items,
                embedding_dim=self.config["embedding_dim"],
                hidden_dims=self.config["hidden_dims"],
                dropout=self.config["dropout"]
            )

        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        return model.to(self.device)

    # ------------------------------------------------
    # Train model
    # ------------------------------------------------
    def train(self, ratings_df):
        print("\n[TrainPipeline] Starting Training Pipeline...\n")

        # 1. Preprocess → encode IDs, normalize ratings, split
        preproc = DataPreprocessor(ratings_df)
        preproc.encode_ids()
        preproc.normalize_ratings()

        train_df, test_df = preproc.split_data()

        # 2. Build datasets + loaders
        train_dataset = RatingsDataset(train_df)
        test_dataset = RatingsDataset(test_df)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config["batch_size"],
            shuffle=True
        )

        # 3. Build model
        model = self.build_model(preproc.num_users, preproc.num_items)

        # 4. Loss function + optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=self.config["learning_rate"])

        # 5. Training loop
        epochs = self.config["epochs"]

        for epoch in range(1, epochs + 1):
            model.train()
            epoch_loss = 0.0

            for user_idx, item_idx, labels in train_loader:
                user_idx = user_idx.to(self.device)
                item_idx = item_idx.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                predictions = model(user_idx, item_idx)
                loss = criterion(predictions, labels)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            print(f"[Epoch {epoch}/{epochs}] Loss: {epoch_loss / len(train_loader):.5f}")

        # 6. Save checkpoint
        os.makedirs("checkpoints", exist_ok=True)
        save_path = "checkpoints/recommender_model.pth"
        torch.save({
            "model_state_dict": model.state_dict(),
            "user2idx": preproc.user2idx,
            "item2idx": preproc.item2idx,
            "num_users": preproc.num_users,
            "num_items": preproc.num_items
        }, save_path)

        print(f"\n[TrainPipeline] Training complete. Model saved → {save_path}")

        return model, preproc
