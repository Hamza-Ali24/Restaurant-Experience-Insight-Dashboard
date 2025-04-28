import streamlit as st
import pandas as pd
import plotly.express as px
import os
import openai
from dotenv import load_dotenv

def show_satisfaction(selected_businesses):
    st.title("📉 MOT Satisfaction")

    # Only allow single business
    if isinstance(selected_businesses, list):
        selected = selected_businesses[0]
    else:
        selected = selected_businesses

    # Load OpenAI API key
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    client = openai.OpenAI(api_key=api_key) if api_key else None

    # Load data
    file = f"ranking/{selected.replace(' ', '_').replace('&', 'and')}_mot_&_sentiment_ranking.csv"
    df = pd.read_csv(file)

    # Filter satisfaction data
    sat = df[df["Type"] == "Satisfaction"].copy().dropna(subset=["Avg_Sentiment"])

    # Sentiment label
    def sentiment_label(score):
        if score == 0.0:
            return "Not Mentioned"
        elif 1.0 <= score < 2.0:
            return "Negative"
        elif 2.0 <= score < 3.0:
            return "Neutral"
        elif score >= 3.0:
            return "Positive"
        else:
            return "Unknown"

    sat["Sentiment_Label"] = sat["Avg_Sentiment"].apply(sentiment_label)

    # Exclude not mentioned
    chart_data = sat[sat["Avg_Sentiment"] > 0.0]

    # Colour map
    colour_map = {
        "Not Mentioned": "#CCCCCC",
        "Negative": "#EF553B",
        "Neutral": "#FECB52",
        "Positive": "#00CC96"
    }

    # Rename for better hover
    chart_data = chart_data.rename(columns={"Avg_Sentiment": "Average Sentiment Score"})

    # Bar chart
    fig = px.bar(
        chart_data.sort_values("Average Sentiment Score", ascending=True),
        x="Average Sentiment Score",
        y="MOT",
        orientation="h",
        color="Sentiment_Label",
        color_discrete_map=colour_map,
        hover_data={
            "MOT": True,
            "Average Sentiment Score": True,
            "Sentiment_Label": False
        }
    )

    fig.update_layout(
        legend_title_text="Sentiment",
        xaxis_title="Average Sentiment Score",
        yaxis_title="Moment of Truth (MoT)",
        xaxis=dict(range=[0, 3]),
        margin=dict(l=20, r=20, t=30, b=20),
        template="plotly"
    )

    st.plotly_chart(fig, use_container_width=True)

    # 🎯 Interpretation
    with st.expander("🎯 How to Interpret the Satisfaction Chart"):
        st.markdown("""
This chart shows **average customer sentiment scores** for each **Moment of Truth (MoT)**.

- **Higher scores (closer to 3.0)** → Happier customers (**Green**)
- **Scores between 2.0 and 3.0** → Neutral or mixed feedback (**Yellow**)
- **Lower scores (closer to 1.0)** → Unhappier customers (**Red**)
- **Grey bars** = Items rarely mentioned (Not Mentioned)

This helps you quickly identify where satisfaction is strong, neutral, or weak.
""")

    # GPT-4 Insight Generator
    st.subheader("💡 Generate Insight from Bar Chart")

    if st.button("Generate Insight") and client:
        with st.spinner("Analysing satisfaction data with GPT-4..."):
            try:
                top = chart_data.sort_values("Average Sentiment Score", ascending=False).head(3)
                bottom = chart_data.sort_values("Average Sentiment Score", ascending=True).head(3)

                top_mots = "; ".join([f"{row['MOT']} ({row['Average Sentiment Score']:.2f})" for _, row in top.iterrows()])
                bottom_mots = "; ".join([f"{row['MOT']} ({row['Average Sentiment Score']:.2f})" for _, row in bottom.iterrows()])

                prompt = (
                    f"For the business '{selected}', the highest-rated Moments of Truth are: {top_mots}.\n"
                    f"The lowest-rated are: {bottom_mots}.\n\n"
                    "Please analyse and provide:\n"
                    "1. What are customers most satisfied with?\n"
                    "2. What are customers least satisfied with?\n"
                    "3. Where should the business focus to improve satisfaction?"
                )

                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a customer experience analyst."},
                        {"role": "user", "content": prompt}
                    ]
                )
                st.success(response.choices[0].message.content)

            except Exception as e:
                st.error(f"Error generating insight: {e}")
