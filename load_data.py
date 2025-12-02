# src/data/load_data.py

import pandas as pd
from pathlib import Path

class DataLoader:
    """
    DataLoader class handles loading raw datasets
    for the NetRecommender system.

    Supports:
    - Ratings data
    - Movie metadata
    - User metadata (optional)
    """

    def __init__(self, data_dir: str = "data/"):
        self.data_dir = Path(data_dir)

    def load_ratings(self, filename: str = "ratings.csv") -> pd.DataFrame:
        """
        Loads user-item interaction data.

        Expected columns:
        - user_id
        - item_id
        - rating
        - timestamp
        """
        file_path = self.data_dir / filename
        self._validate_exists(file_path)

        df = pd.read_csv(file_path)
        self._validate_columns(
            df, ["user_id", "item_id", "rating", "timestamp"]
        )

        print(f"[DataLoader] Loaded ratings: {df.shape}")
        return df

    def load_movies(self, filename: str = "movies.csv") -> pd.DataFrame:
        """
        Loads movie or item metadata.

        Expected columns:
        - item_id
        - title
        - genres
        """
        file_path = self.data_dir / filename
        self._validate_exists(file_path)

        df = pd.read_csv(file_path)
        self._validate_columns(
            df, ["item_id", "title", "genres"]
        )

        print(f"[DataLoader] Loaded movie metadata: {df.shape}")
        return df

    def load_users(self, filename: str = "users.csv") -> pd.DataFrame:
        """
        Optional: load user metadata.
        """
        file_path = self.data_dir / filename
        if not file_path.exists():
            print(f"[DataLoader] No user metadata found at {file_path}. Skipping.")
            return pd.DataFrame()

        df = pd.read_csv(file_path)
        print(f"[DataLoader] Loaded user metadata: {df.shape}")
        return df

    # ---------------------------
    # Helper validation functions
    # ---------------------------

    def _validate_exists(self, path: Path):
        if not path.exists():
            raise FileNotFoundError(f"[DataLoader] File not found: {path}")

    def _validate_columns(self, df: pd.DataFrame, required_cols: list):
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(
                f"[DataLoader] Missing required columns: {missing}"
            )
