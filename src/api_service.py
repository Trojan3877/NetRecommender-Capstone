from fastapi import FastAPI, Query
from typing import List

app = FastAPI(title="NetRecommender API")

@app.get("/recommend")
def recommend(user_id: int = Query(...), k: int = 10) -> List[int]:
    # TODO: replace with model-based recs
    return [101, 102, 103][:k]
