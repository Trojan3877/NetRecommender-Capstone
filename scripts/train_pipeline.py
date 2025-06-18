import pandas as pd
from recommender.python.hybrid_recommender import HybridRecommender

def load_data():
    """
    Loads users and movies data from local CSV files.
    Returns: (movies_df, users_df)
    """
    movies_df = pd.read_csv("data/movies.csv")
    users_df = pd.read_csv("data/users.csv")
    return movies_df, users_df

def train_and_recommend():
    """
    Initializes the recommender and outputs top-N recommendations.
    """
    movies_df, users_df = load_data()
    recommender = HybridRecommender(movies_df, users_df)

    for uid in users_df['userId']:
        recommendations = recommender.recommend(user_id=uid, top_n=5)
        print(f"User {uid} Recommendations: {recommendations}")

if __name__ == "__main__":
    train_and_recommend()
