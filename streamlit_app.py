import streamlit as st
import os
from app_pages.quadrant import show_quadrant
from app_pages.satisfaction import show_satisfaction
from app_pages.importance import show_importance
from app_pages.compare import show_comparison

# Set Streamlit page config
st.set_page_config(page_title="Restaurant Experience Insight Dashboard", layout="wide")

# Sidebar Title
st.sidebar.title("Restaurant Experience Insight Dashboard")

# Load business files
ranking_dir = "ranking"
files = [f for f in os.listdir(ranking_dir) if f.endswith(".csv")]
business_names = [f.replace("_mot_&_sentiment_ranking.csv", "").replace("_", " ").replace("and", "&") for f in files]

# Sidebar - Page Selection
page = st.sidebar.radio("Select a Page", [
    "📊 MOT Priority Matrix (Scatter Plot)",
    "📉 MOT Satisfaction (Bar Chart)",
    "📈 MOT Importance (Bar Chart)",
    "📡 Compare Businesses (Radar Chart)"
])

# Sidebar - Business Selection based on Page
if page == "📡 Compare Businesses (Radar Chart)":
    selected_businesses = st.sidebar.multiselect(
        "Select Businesses (2-3 Required)",
        business_names,
        default=business_names[:2]
    )

    if len(selected_businesses) < 2:
        st.sidebar.error("Please select at least two businesses to compare.")
        st.stop()

else:
    selected_businesses = st.sidebar.selectbox(
        "Select a Business",
        business_names
    )

# Load the correct page
if page == "📊 MOT Priority Matrix (Scatter Plot)":
    show_quadrant(selected_businesses)

elif page == "📉 MOT Satisfaction (Bar Chart)":
    show_satisfaction(selected_businesses)

elif page == "📈 MOT Importance (Bar Chart)":
    show_importance(selected_businesses)

elif page == "📡 Compare Businesses (Radar Chart)":
    show_comparison(selected_businesses)
