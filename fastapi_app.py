# src/api/fastapi_app.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

from src.pipelines.inference_pipeline import InferencePipeline


# -----------------------------------------------------
# FastAPI Application Setup
# -----------------------------------------------------
app = FastAPI(
    title="NetRecommender API",
    description="Production-grade API for Neural Collaborative Filtering recommender system.",
    version="1.0.0"
)

# Load model checkpoint once at startup
pipeline = InferencePipeline(checkpoint_path="checkpoints/recommender_model.pth")


# -----------------------------------------------------
# Request Model (Optional Body Input)
# -----------------------------------------------------
class RecommendRequest(BaseModel):
    top_k: int = 10


# -----------------------------------------------------
# Health Check Endpoint
# -----------------------------------------------------
@app.get("/health")
def health_check():
    return {"status": "ok", "message": "NetRecommender API is running."}


# -----------------------------------------------------
# Recommendation Endpoint
# -----------------------------------------------------
@app.get("/recommend/{user_id}")
def recommend_user(user_id: str, top_k: int = 10):
    """
    Path Parameters:
        user_id (str): The user ID to generate recommendations for
        top_k (int): Number of items to recommend

    Returns:
        JSON containing top_k recommended items + scores
    """
    try:
        recommended_items, scores = pipeline.recommend(user_id, top_k)
        return {
            "user_id": user_id,
            "top_k": top_k,
            "recommendations": [
                {"item_id": item_id, "score": float(score)}
                for item_id, score in zip(recommended_items, scores)
            ]
        }

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# -----------------------------------------------------
# Optional Local Server Run
# -----------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )
