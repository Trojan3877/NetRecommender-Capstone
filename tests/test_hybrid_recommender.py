import pytest
from assistant.hybrid_recommender import HybridRecommender

@pytest.fixture
def recommender():
    return HybridRecommender()

def test_recommend_valid_user(recommender):
    user_id = 1
    recommendations = recommender.recommend(user_id=user_id, top_n=5, alpha=0.5)
    
    assert isinstance(recommendations, list)
    assert len(recommendations) <= 5
    for rec in recommendations:
        assert isinstance(rec, int)  # Assuming movie IDs are integers

def test_invalid_user_id(recommender):
    with pytest.raises(ValueError):
        recommender.recommend(user_id=-1, top_n=5)

def test_invalid_top_n(recommender):
    with pytest.raises(ValueError):
        recommender.recommend(user_id=1, top_n=-10)

def test_alpha_range(recommender):
    with pytest.raises(ValueError):
        recommender.recommend(user_id=1, top_n=5, alpha=1.5)
