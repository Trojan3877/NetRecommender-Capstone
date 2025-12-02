# src/evaluation/metrics.py

import numpy as np


def precision_at_k(recommended_items, true_items, k=10):
    """
    Precision@K = (# of recommended items in top K that are relevant) / K
    """
    recommended_k = recommended_items[:k]
    relevant = set(recommended_k).intersection(set(true_items))
    return len(relevant) / k


def recall_at_k(recommended_items, true_items, k=10):
    """
    Recall@K = (# of relevant items in top K) / (# of relevant items total)
    """
    recommended_k = recommended_items[:k]
    relevant = set(recommended_k).intersection(set(true_items))
    return len(relevant) / max(len(true_items), 1)


def hit_rate_at_k(recommended_items, true_items, k=10):
    """
    Hit Rate@K = 1 if ANY relevant item is in top K, else 0
    """
    recommended_k = set(recommended_items[:k])
    return 1 if recommended_k.intersection(set(true_items)) else 0


def dcg_at_k(scores, k=10):
    """
    Discounted Cumulative Gain
    """
    scores = np.asfarray(scores)[:k]
    return np.sum(scores / np.log2(np.arange(2, len(scores) + 2)))


def ndcg_at_k(recommended_items, true_items, k=10):
    """
    NDCG@K = DCG@K / IDCG@K
    """
    # 1 for relevant, 0 for non-relevant
    relevance = [1 if item in true_items else 0 for item in recommended_items]

    dcg = dcg_at_k(relevance, k)
    ideal_relevance = sorted(relevance, reverse=True)
    idcg = dcg_at_k(ideal_relevance, k)

    return dcg / idcg if idcg > 0 else 0.0


def average_precision_at_k(recommended_items, true_items, k=10):
    """
    Mean Average Precision@K (MAP@K)
    """
    score = 0.0
    hits = 0

    for idx, item in enumerate(recommended_items[:k]):
        if item in true_items:
            hits += 1
            score += hits / (idx + 1)

    return score / max(min(len(true_items), k), 1)
