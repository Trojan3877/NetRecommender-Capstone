# src/data/preprocess.py

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix


class DataPreprocessor:
    """
    DataPreprocessor handles splitting, encoding, and transforming
    raw interactions for recommender system models.

    Outputs:
        - Encoded user_ids and item_ids
        - Train/Test splits
        - Sparse interaction matrix
        - Lookup dictionaries for embeddings
    """

    def __init__(self, ratings_df: pd.DataFrame):
        self.raw_df = ratings_df.copy()

        self.user2idx = {}
        self.idx2user = {}
        self.item2idx = {}
        self.idx2item = {}

        self.num_users = 0
        self.num_items = 0

    # ---------------------------------------------------------
    # 1. Encode IDs → sequential indices for embedding layers
    # ---------------------------------------------------------
    def encode_ids(self):
        """
        Creates mapping:
        user_id → index
        item_id → index
        """

        unique_users = self.raw_df["user_id"].unique()
        unique_items = self.raw_df["item_id"].unique()

        self.user2idx = {uid: i for i, uid in enumerate(unique_users)}
        self.idx2user = {i: uid for uid, i in self.user2idx.items()}

        self.item2idx = {iid: i for i, iid in enumerate(unique_items)}
        self.idx2item = {i: iid for iid, i in self.item2idx.items()}

        self.raw_df["user_idx"] = self.raw_df["user_id"].map(self.user2idx)
        self.raw_df["item_idx"] = self.raw_df["item_id"].map(self.item2idx)

        self.num_users = len(unique_users)
        self.num_items = len(unique_items)

        print(f"[Preprocessor] Encoded {self.num_users} users, {self.num_items} items")

    # ---------------------------------------------------------
    # 2. Train/Test Split
    # ---------------------------------------------------------
    def split_data(self, test_size=0.2, seed=42):
        train_df, test_df = train_test_split(
            self.raw_df,
            test_size=test_size,
            random_state=seed,
            shuffle=True
        )
        print(f"[Preprocessor] Train size: {train_df.shape}, Test size: {test_df.shape}")
        return train_df, test_df

    # ---------------------------------------------------------
    # 3. Create Sparse Interaction Matrix (for CF models)
    # ---------------------------------------------------------
    def build_interaction_matrix(self):
        """
        Builds user-item sparse matrix for collaborative filtering.

        Matrix shape: (num_users, num_items)
        """
        rows = self.raw_df["user_idx"].values
        cols = self.raw_df["item_idx"].values
        ratings = self.raw_df["rating"].values

        interaction_matrix = csr_matrix(
            (ratings, (rows, cols)),
            shape=(self.num_users, self.num_items)
        )

        print(f"[Preprocessor] Built sparse interaction matrix: {interaction_matrix.shape}")
        return interaction_matrix

    # ---------------------------------------------------------
    # 4. Normalize Ratings (for neural models)
    # ----------------------------------
