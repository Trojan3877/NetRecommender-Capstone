# src/models/baseline_cf.py

import torch
import torch.nn as nn


class MatrixFactorization(nn.Module):
    """
    Baseline Collaborative Filtering model using Matrix Factorization (MF).

    Features:
    - User and Item Embeddings
    - User Bias & Item Bias
    - Global Bias
    - Optional Sigmoid activation (for binary/implicit data)
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        use_sigmoid: bool = False
    ):
        super().__init__()

        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)

        # Bias terms
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))

        self.use_sigmoid = use_sigmoid

        self._init_weights()

    def _init_weights(self):
        """
        Xavier initialization for better training stability.
        """
        nn.init.xavier_uniform_(self.user_embeddings.weight)
        nn.init.xavier_uniform_(self.item_embeddings.weight)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, user_idx: torch.Tensor, item_idx: torch.Tensor):
        """
        Predicts rating or interaction score.

        Args:
            user_idx: Tensor [batch]
            item_idx: Tensor [batch]

        Returns:
            predictions: Tensor [batch]
        """

        user_vec = self.user_embeddings(user_idx)      # [batch, dim]
        item_vec = self.item_embeddings(item_idx)      # [batch, dim]

        # Dot-product similarity
        dot = (user_vec * item_vec).sum(dim=1)

        # Add bias terms
        user_b = self.user_bias(user_idx).squeeze()
        item_b = self.item_bias(item_idx).squeeze()

        predictions = dot + user_b + item_b + self.global_bias

        if self.use_sigmoid:
            predictions = torch.sigmoid(predictions)

        return predictions
