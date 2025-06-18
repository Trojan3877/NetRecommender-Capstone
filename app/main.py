from fastapi import FastAPI, Query
from pydantic import BaseModel
from assistant.hybrid_recommender import HybridRecommender

app = FastAPI(title="Netflix Recommender API")

# Initialize model (mock: you would load or train your model here)
recommender = HybridRecommender()

class RecommendationRequest(BaseModel):
    user_id: int
    top_n: int = 10
    alpha: float = 0.5  # Weighting between content and collaborative filtering

@app.get("/")
def read_root():
    return {"message": "Welcome to the Netflix Recommender API!"}

@app.post("/recommend")
def get_recommendation(request: RecommendationRequest):
    results = recommender.recommend(user_id=request.user_id, top_n=request.top_n, alpha=request.alpha)
    return {"user_id": request.user_id, "recommendations": results}
