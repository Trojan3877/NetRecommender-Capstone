# main.py

import pandas as pd

from src.pipelines.train_pipeline import TrainingPipeline
from src.pipelines.inference_pipeline import InferencePipeline
from src.evaluation.ranking_eval import RankingEvaluator


def load_data(path="data/ratings.csv"):
    """
    Load user-item-rating interactions.
    """
    print("[Main] Loading dataset...")
    return pd.read_csv(path)


def run_training():
    """
    Runs the full model training pipeline.
    """
    # 1. Load dataset
    df = load_data()

    # 2. Initialize training pipeline
    trainer = TrainingPipeline(config_path="configs/train_config.yaml")

    # 3. Train model
    model, preproc = trainer.train(df)

    return model, preproc


def run_evaluation(preproc, top_k=10):
    """
    Optional evaluation step after training.
    """
    print("\n[Main] Running evaluation...")

    # Prepare ground truth dict for ranking metrics
    ground_truth = (
        preproc.ratings_df.groupby("user_id")["item_id"].apply(list).to_dict()
    )

    # Initialize inference pipeline
    pipeline = InferencePipeline(checkpoint_path="checkpoints/recommender_model.pth")

    # Generate predictions for all users
    predictions = {}
    for user_id in list(ground_truth.keys())[:200]:  # limit for speed
        items, scores = pipeline.recommend(user_id=user_id, top_k=top_k)
        predictions[user_id] = (items, scores)

    # Evaluate
    evaluator = RankingEvaluator(top_k=top_k)
    metrics = evaluator.evaluate_batch(predictions, ground_truth)

    evaluator.print_results(metrics)

    return metrics


if __name__ == "__main__":
    print("==========================================")
    print("         NetRecommender Pipeline")
    print("==========================================")

    # Train model first
    model, preproc = run_training()

    # Optional: Evaluate model
    run_evaluation(preproc)

    print("\n[Main] Training + Evaluation complete.")
    print("[Main] Ready for FastAPI deployment using Docker!")
