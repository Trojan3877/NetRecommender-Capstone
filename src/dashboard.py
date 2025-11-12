import streamlit as st
import os

st.set_page_config(page_title="NetRecommender Dashboard", layout="wide")
st.title("📊 NetRecommender - Metrics")
st.write("This page renders snapshots from `tracking/metrics.md` and links to MLflow.")

if os.path.exists("tracking/metrics.md"):
    with open("tracking/metrics.md") as f:
        st.markdown(f.read())
else:
    st.info("No metrics yet. Trigger the Airflow DAG and refresh.")
st.link_button("Open MLflow", "http://localhost:5000")
