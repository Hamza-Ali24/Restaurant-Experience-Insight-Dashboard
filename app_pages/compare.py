import streamlit as st
import pandas as pd
import os
import plotly.express as px
import openai
from dotenv import load_dotenv

def show_comparison(selected_businesses):
    st.title("📡 Compare Businesses")

    # Must be 2 or more businesses
    if not isinstance(selected_businesses, list) or len(selected_businesses) < 2:
        st.error("Please select at least two businesses to compare.")
        return

    # Load OpenAI API key
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    client = openai.OpenAI(api_key=api_key) if api_key else None

    # Load and combine ranking data
    ranking_dir = "ranking"
    dfs = []
    for business in selected_businesses:
        file = f"{ranking_dir}/{business.replace(' ', '_').replace('&', 'and')}_mot_&_sentiment_ranking.csv"
        df = pd.read_csv(file)
        df = df[df["Type"] == "Satisfaction"].copy()
        df["Business"] = business
        dfs.append(df)

    combined = pd.concat(dfs)

    # Fixed order for MOT on Radar Chart
    MOT_CATEGORIES = [
        "Arrival & First Impressions", "Waiting Time", "Ambience & Atmosphere",
        "Service Interaction", "Menu Presentation & Ordering", "Food & Drink Arrival Time",
        "Food Quality & Presentation", "Handling of Dietary Requirements", "Toilet Cleanliness & Maintenance",
        "Billing & Payment Process", "Issue Resolution & Complaint Handling", "Word-of-Mouth & Recommendations"
    ]
     # Prepare data for plotting
    radar_df = (
        combined.pivot(index="MOT", columns="Business", values="Avg_Sentiment")
        .reindex(MOT_CATEGORIES)
        .reset_index()
    )

    plot_df = pd.melt(radar_df, id_vars="MOT", var_name="Business", value_name="Avg_Sentiment")

    # Create Radar chart
    fig = px.line_polar(
        plot_df,
        r="Avg_Sentiment",
        theta="MOT",
        color="Business",
        line_close=True
    )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 3],
                tickvals=[0, 1, 2, 3],
                tickfont=dict(size=10)
            )
        ),
        template="plotly",
        legend_itemclick=False,         # Prevent clicking legend items to toggle visibility
        legend_itemdoubleclick=False    # Prevent isolating a line on double-click
    )
     # Display Radar Chart
    st.plotly_chart(fig, use_container_width=True)

    # 🎯 Interpretation guide
    with st.expander("🎯 How to Interpret the Radar Chart"):
        st.markdown("""
This radar chart shows **customer satisfaction** across key **Moments of Truth (MoTs)** for each selected business.

- **Closer to the edge (score 3.0)** → Higher customer satisfaction  
- **Closer to the centre (score 1.0)** → Lower customer satisfaction  

Use this to spot strengths and weaknesses for each business at a glance.
""")

    # GPT-4 Insight Generator
    st.subheader("💡 Generate Insight from Radar Chart")

    if st.button("Generate Insight") and client:
        with st.spinner("Analysing radar chart data with GPT-4..."):
            try:
                # Prepare summarised data for GPT
                gpt_df = (
                    combined.pivot(index="MOT", columns="Business", values="Avg_Sentiment")
                    .reindex(MOT_CATEGORIES)
                    .round(2)
                )
                mot_lines = []
                for mot, row in gpt_df.iterrows():
                    scores = ", ".join([f"{biz}: {score}" for biz, score in row.items()])
                    mot_lines.append(f"{mot} → {scores}")
                radar_summary = "\n".join(mot_lines)
                 #Build prompt
                prompt = (
                    "You are analysing customer satisfaction across multiple businesses based on key Moments of Truth (MoTs).\n\n"
                    f"Here are the average sentiment scores:\n"
                    f"{radar_summary}\n\n"
                    "Please answer:\n"
                    "1. Which businesses are leading in satisfaction overall?\n"
                    "2. What are the major strengths and weaknesses for each business?\n"
                    "3. What strategic improvements would you suggest for each?"
                )
                  # Call OpenAI GPT model
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a customer experience strategist."},
                        {"role": "user", "content": prompt}
                    ]
                )
                 # Display GPT-4 mini Insight
                st.success(response.choices[0].message.content)

            except Exception as e:
                st.error(f"Error generating insight: {e}")
