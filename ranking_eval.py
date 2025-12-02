# src/evaluation/ranking_eval.py

import numpy as np
from src.evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    ndcg_at_k,
    hit_rate_at_k,
    average_precision_at_k
)


class RankingEvaluator:
    """
    Unified evaluator for recommender system ranking metrics.

    Supports:
        - Precision@K
        - Recall@K
        - NDCG@K
        - MAP@K
        - HitRate@K

    Works with:
        - Neural CF model outputs
        - Matrix Factorization outputs
        - InferencePipeline recommend() output
    """

    def __init__(self, top_k=10):
        self.top_k = top_k

    def evaluate_user(self, recommended_items, true_items):
        """
        Evaluate ranking metrics for a single user.

        Args:
            recommended_items: list of item_ids ranked by model
            true_items: list of items the user truly interacted with
        """

        return {
            f"precision@{self.top_k}": precision_at_k(recommended_items, true_items, self.top_k),
            f"recall@{self.top_k}": recall_at_k(recommended_items, true_items, self.top_k),
            f"ndcg@{self.top_k}": ndcg_at_k(recommended_items, true_items, self.top_k),
            f"map@{self.top_k}": average_precision_at_k(recommended_items, true_items, self.top_k),
            f"hitrate@{self.top_k}": hit_rate_at_k(recommended_items, true_items, self.top_k)
        }

    def evaluate_batch(self, predictions_dict, ground_truth_dict):
        """
        Batch evaluation across many users.

        Args:
            predictions_dict: {user_id: ([recommended_items], [scores])}
            ground_truth_dict: {user_id: [true_items]}

        Returns:
            aggregated_metrics: dict with averaged metrics
        """

        metrics_list = []

        for user_id, (recommended_items, _) in predictions_dict.items():
            if user_id not in ground_truth_dict:
                continue

            user_true_items = ground_truth_dict[user_id]
            user_metrics = self.evaluate_user(recommended_items, user_true_items)
            metrics_list.append(user_metrics)

        # Aggregate metrics across users
        aggregated = {}
        if not metrics_list:
            return None

        for metric in metrics_list[0].keys():
            aggregated[metric] = float(np.mean([m[metric] for m in metrics_list]))

        return aggregated

    def print_results(self, aggregated_metrics):
        """
        Nicely prints the evaluation results.
        """
        if aggregated_metrics is None:
            print("[RankingEvaluator] No metrics to print.")
            return

        print("\n[RankingEvaluator] Final Ranking Metrics")
        print("----------------------------------------")
        for metric, value in aggregated_metrics.items():
            print(f"{metric}: {value:.4f}")
