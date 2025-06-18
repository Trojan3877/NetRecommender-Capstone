import streamlit as st
from assistant.hybrid_recommender import HybridRecommender

st.set_page_config(page_title="Netflix Recommender Dashboard", layout="wide")

st.title("🎬 Netflix-Style Recommendation Engine")
st.markdown("This interactive dashboard allows you to generate hybrid recommendations for a given user ID.")

recommender = HybridRecommender()

# User inputs
user_id = st.number_input("Enter User ID", min_value=1, max_value=10000, value=1, step=1)
top_n = st.slider("Number of Recommendations", 5, 20, 10)
alpha = st.slider("Alpha (blend content/collab)", 0.0, 1.0, 0.5)

if st.button("Get Recommendations"):
    try:
        recommendations = recommender.recommend(user_id=int(user_id), top_n=top_n, alpha=alpha)

        # Display
        st.success(f"Top {top_n} Recommendations for User {user_id}")
        st.table({
            "Rank": list(range(1, len(recommendations)+1)),
            "Movie ID": recommendations
        })
    except Exception as e:
        st.error(f"Failed to generate recommendations: {str(e)}")
