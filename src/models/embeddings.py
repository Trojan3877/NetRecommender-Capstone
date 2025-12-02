# src/models/embeddings.py

import torch
import torch.nn as nn


class EmbeddingLayer(nn.Module):
    """
    Generic Embedding Layer for Users and Items.

    This layer is used in:
    - Matrix Factorization
    - Neural Collaborative Filtering (NCF)
    - Hybrid Recommenders
    - Future Transformer or Attention-based models
    """

    def __init__(
        self,
        num_entities: int,
        embedding_dim: int = 64,
        dropout: float = 0.1,
        layer_norm: bool = False
    ):
        super().__init__()

        self.embedding = nn.Embedding(num_entities, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embedding_dim) if layer_norm else None

        self._init_weights()

    def _init_weights(self):
        """
        Xavier initialization for more stable training.
        """
        nn.init.xavier_uniform_(self.embedding.weight)

    def forward(self, indices: torch.Tensor):
        """
        Inputs:
            indices: Tensor of shape [batch]
        Returns:
            Embeddings of shape [batch, embedding_dim]
        """
        x = self.embedding(indices)
        x = self.dropout(x)

        if self.layer_norm:
            x = self.layer_norm(x)

        return x


class UserItemEmbeddings(nn.Module):
    """
    Combined wrapper for user + item embeddings.

    Provides:
    - user_embedding_layer
    - item_embedding_layer
    - forward() returns both embeddings
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int = 64,
        dropout: float = 0.1,
        layer_norm: bool = False
    ):
        super().__init__()

        self.user_embeddings = EmbeddingLayer(
            num_entities=num_users,
            embedding_dim=embedding_dim,
            dropout=dropout,
            layer_norm=layer_norm
        )

        self.item_embeddings = EmbeddingLayer(
            num_entities=num_items,
            embedding_dim=embedding_dim,
            dropout=dropout,
            layer_norm=layer_norm
        )

    def forward(self, user_idx: torch.Tensor, item_idx: torch.Tensor):
        """
        Return:
            user_vector: [batch, embedding_dim]
            item_vector: [batch, embedding_dim]
        """
        user_vector = self.user_embeddings(user_idx)
        item_vector = self.item_embeddings(item_idx)
        return user_vector, item_vector
