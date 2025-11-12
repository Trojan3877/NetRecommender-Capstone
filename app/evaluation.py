from metrixflow import MetricsTracker

tracker = MetricsTracker(experiment_name="NetRecommender-Capstone")
tracker.log_metrics({
    "RMSE": 0.93,
    "Precision@10": 0.81,
    "Recall@10": 0.77,
    "MAP": 0.69
})
tracker.save("tracking/metrics.md")
