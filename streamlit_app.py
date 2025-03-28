import streamlit as st
import pandas as pd
import os

# Set Streamlit page settings
st.set_page_config(page_title="MoT Rankings Dashboard", layout="wide")

# Title and description
st.title("Moments of Truth (MoT) Rankings Dashboard")
st.markdown("""
This dashboard shows **Importance (Mentions)** and **Satisfaction (Sentiment)** rankings for each restaurant based on customer reviews.
Select a restaurant from the dropdown to view their prioritised Moments of Truth.
""")

# Directory where ranking CSVs are saved
ranking_dir = "ranking"

# Get all ranking files
csv_files = [f for f in os.listdir(ranking_dir) if f.endswith(".csv")]

# Create a display-friendly name list
restaurant_names = [
    f.replace("_mot_&_sentiment_ranking.csv", "")
     .replace("_", " ")
     .replace("and", "&")
     for f in csv_files
]

# Restaurant selection
selected_restaurant = st.selectbox("Choose a Restaurant:", restaurant_names)

# Reconstruct filename from selected option
filename = f"{selected_restaurant.replace(' ', '_').replace('&', 'and')}_mot_&_sentiment_ranking.csv"
file_path = os.path.join(ranking_dir, filename)

# Load the CSV
df = pd.read_csv(file_path)

# Split DataFrames by Type
importance_df = df[df["Type"] == "Importance"][["MOT", "Mentions"]].sort_values(by="Mentions", ascending=False).reset_index(drop=True)
satisfaction_df = df[df["Type"] == "Satisfaction"][["MOT", "Avg_Sentiment"]].sort_values(by="Avg_Sentiment", ascending=True).reset_index(drop=True)

# Layout in two columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("Importance Ranking (Mentions)")
    st.dataframe(importance_df, use_container_width=True)

with col2:
    st.subheader("Satisfaction Ranking (Avg Sentiment)")
    st.dataframe(satisfaction_df, use_container_width=True)
