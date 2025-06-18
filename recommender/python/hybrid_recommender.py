import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import numpy as np

class HybridRecommender:
    def __init__(self, movies_df, users_df):
        self.movies_df = movies_df
        self.users_df = users_df
        self.movie_genre_vectors = self._create_genre_vectors()
        self.user_rating_matrix = self._create_rating_matrix()

    def _create_genre_vectors(self):
        genres = self.movies_df['genres'].str.get_dummies(sep='|')
        return genres.values

    def _create_rating_matrix(self):
        user_dict = {}
        for _, row in self.users_df.iterrows():
            user_ratings = row['rated_movies'].split('|')
            ratings = {}
            for item in user_ratings:
                mid, score = item.split(':')
                ratings[int(mid)] = float(score)
            user_dict[row['userId']] = ratings
        return user_dict

    def _collaborative_score(self, user_id):
        target_ratings = self.user_rating_matrix.get(user_id, {})
        similarity_scores = defaultdict(float)

        for other_user, ratings in self.user_rating_matrix.items():
            if other_user == user_id:
                continue

            common = set(target_ratings.keys()) & set(ratings.keys())
            if not common:
                continue

            vec1 = np.array([target_ratings[m] for m in common])
            vec2 = np.array([ratings[m] for m in common])

            sim = np.corrcoef(vec1, vec2)[0][1] if len(vec1) > 1 else 0
            for m, r in ratings.items():
                if m not in target_ratings:
                    similarity_scores[m] += sim * r

        return similarity_scores

    def _content_score(self, user_id):
        user_rated = self.user_rating_matrix.get(user_id, {})
        watched_ids = list(user_rated.keys())
        if not watched_ids:
            return {}

        watched_vec = self.movie_genre_vectors[[m - 1 for m in watched_ids]]
        avg_vec = watched_vec.mean(axis=0).reshape(1, -1)

        sim_scores = cosine_similarity(self.movie_genre_vectors, avg_vec).flatten()
        return {i + 1: score for i, score in enumerate(sim_scores) if (i + 1) not in watched_ids}

    def recommend(self, user_id, top_n=5, alpha=0.5):
        collab = self._collaborative_score(user_id)
        content = self._content_score(user_id)
        combined_scores = defaultdict(float)

        for movie_id in set(collab.keys()).union(content.keys()):
            combined_scores[movie_id] = alpha * collab.get(movie_id, 0) + (1 - alpha) * content.get(movie_id, 0)

        sorted_movies = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        return [mid for mid, _ in sorted_movies[:top_n]]
