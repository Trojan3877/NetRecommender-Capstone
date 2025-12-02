# src/pipelines/inference_pipeline.py

import torch
import numpy as np

from src.models.baseline_cf import MatrixFactorization
from src.models.neural_recommender import NeuralRecommender


class InferencePipeline:
    """
    Loads a trained model checkpoint and generates recommendations.

    Supports both:
    - Matrix Factorization
    - Neural CF (Deep Learning)
    """

    def __init__(self, checkpoint_path="checkpoints/recommender_model.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Extract metadata
        self.num_users = checkpoint["num_users"]
        self.num_items = checkpoint["num_items"]
        self.user2idx = checkpoint["user2idx"]
        self.item2idx = checkpoint["item2idx"]

        # Choose model type by embedding size
        # (Simple but effective heuristic)
        embedding_dim = list(checkpoint["model_state_dict"].values())[0].shape[1]

        if embedding_dim == 64:  # Could also store model_type in checkpoint
            self.model = NeuralRecommender(
                num_users=self.num_users,
                num_items=self.num_items,
                embedding_dim=64
            )
        else:
            self.model = MatrixFactorization(
                num_users=self.num_users,
                num_items=self.num_items,
                embedding_dim=embedding_dim
            )

        # Load weights
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        print(f"[Inference] Model loaded from {checkpoint_path}")

    # ------------------------------------------------
    # Predict a single user-item score
    # ------------------------------------------------
    def predict(self, user_idx, item_idx):
        user_idx = torch.tensor([user_idx], dtype=torch.long).to(self.device)
        item_idx = torch.tensor([item_idx], dtype=torch.long).to(self.device)

        with torch.no_grad():
            score = self.model(user_idx, item_idx).cpu().item()

        return score

    # ------------------------------------------------
    # Generate Top-K recommendations for a user
    # ------------------------------------------------
    def recommend(self, user_id, top_k=10):
        if user_id not in self.user2idx:
            raise ValueError(f"User ID {user_id} not found in training set")

        user_idx = self.user2idx[user_id]

        # Predict for all items
        all_item_indices = list(self.item2idx.values())
        all_item_tensor = torch.tensor(all_item_indices, dtype=torch.long).to(self.device)
        user_tensor = torch.tensor([user_idx] * len(all_item_indices), dtype=torch.long).to(self.device)

        with torch.no_grad():
            scores = self.model(user_tensor, all_item_tensor).cpu().numpy()

        # Rank items by score
        top_indices = np.argsort(scores)[::-1][:top_k]

        # Map back to item IDs
        inv_item = {v: k for k, v in self.item2idx.items()}
        top_items = [inv_item[idx] for idx in top_indices]

        return top_items, scores[top_indices]

    # ------------------------------------------------
    # Batch recommend for multiple users
    # ------------------------------------------------
    def batch_recommend(self, user_ids, top_k=10):
        return {uid: self.recommend(uid, top_k) for uid in user_ids}
