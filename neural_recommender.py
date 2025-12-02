# src/models/neural_recommender.py

import torch
import torch.nn as nn
from src.models.embeddings import UserItemEmbeddings


class NeuralRecommender(nn.Module):
    """
    Neural Collaborative Filtering (NCF) model.

    Architecture:
        - User & Item Embedding Layers
        - Concatenation of embeddings
        - Multi-Layer Perceptron (MLP)
        - Sigmoid output (for ranking/probability)
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        hidden_dims: list = [128, 64, 32],
        dropout: float = 0.2
    ):
        super().__init__()

        # Shared embedding backbone
        self.embedding_layer = UserItemEmbeddings(
            num_users=num_users,
            num_items=num_items,
            embedding_dim=embedding_dim,
            dropout=dropout,
            layer_norm=True  # L6 upgrade
        )

        # Build the MLP tower
        layers = []
        input_dim = embedding_dim * 2  # user + item concatenation

        for h_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            layers.append(nn.LayerNorm(h_dim))
            input_dim = h_dim

        self.mlp = nn.Sequential(*layers)

        # Final prediction layer
        self.output_layer = nn.Linear(input_dim, 1)

        # Sigmoid output for probability
        self.activation = nn.Sigmoid()

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """
        Xavier initialization across layers.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, user_idx: torch.Tensor, item_idx: torch.Tensor):
        """
        Args:
            user_idx: [batch]
            item_idx: [batch]

        Returns:
            Probability that user will interact with item → [batch]
        """

        # 1. Get embeddings
        user_vec, item_vec = self.embedding_layer(user_idx, item_idx)

        # 2. Concatenate user + item
        x = torch.cat([user_vec, item_vec], dim=1)

        # 3. MLP tower
        x = self.mlp(x)

        # 4. Final prediction
        x = self.output_layer(x)

        # 5. Sigmoid score
        score = self.activation(x).squeeze()

        return score
