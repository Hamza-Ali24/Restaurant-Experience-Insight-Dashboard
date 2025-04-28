import streamlit as st
import pandas as pd
import plotly.express as px
import os
import openai
from dotenv import load_dotenv

def show_importance(selected_businesses):
    st.title("📈 MOT Importance")

    # Only allow single business
    if isinstance(selected_businesses, list):
        selected = selected_businesses[0]
    else:
        selected = selected_businesses

    # Load OpenAI API key
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    client = openai.OpenAI(api_key=api_key) if api_key else None

    # Load business data
    file = f"ranking/{selected.replace(' ', '_').replace('&', 'and')}_mot_&_sentiment_ranking.csv"
    df = pd.read_csv(file)

    # Filter for importance data
    imp = df[df["Type"] == "Importance"]

    # Create horizontal Bar chart
    fig = px.bar(
        imp.sort_values("Mentions", ascending=False),
        x="Mentions",
        y="MOT",
        orientation="h",
        color="MOT",
        color_discrete_sequence=px.colors.qualitative.Set2
    )

    fig.update_layout(
        showlegend=False,
        xaxis_title="Mentions (Importance)",
        yaxis_title="Moment of Truth (MoT)",
        margin=dict(l=20, r=20, t=30, b=20),
        template="plotly"
    )

    st.plotly_chart(fig, use_container_width=True)

    # 🎯 Interpretation guide
    with st.expander("🎯 How to Interpret the Importance Chart"):
        st.markdown("""
This chart shows the **importance** of each **Moment of Truth (MoT)** based on how often customers mentioned them.

- **More mentions** → Higher perceived importance
- **Fewer mentions** → Lower perceived importance

Use this to prioritise which interactions matter most to your customers.
""")

    # GPT-4 Insight Generator
    st.subheader("💡 Generate Insight from Bar Chart")

    if st.button("Generate Insight") and client:
        with st.spinner("Analysing importance data with GPT-4..."):
            try:
                # Focus on the top 5 most mentioned MOTs
                top = imp.sort_values("Mentions", ascending=False).head(5)

                mot_lines = [f"{row['MOT']} ({row['Mentions']})" for _, row in top.iterrows()]
                mot_summary = "; ".join(mot_lines)
                 # Build Prompt
                prompt = (
                    f"For the business '{selected}', the top Moments of Truth based on mentions are: {mot_summary}.\n\n"
                    "Please analyse and provide:\n"
                    "1. What do customers seem to value the most?\n"
                    "2. Which areas should be prioritised for improvement or continued focus?"
                )
                   #Call OpenAI GPT model
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a customer experience consultant."},
                        {"role": "user", "content": prompt}
                    ]
                )
                 # Display GPT-4 mini Insight
                st.success(response.choices[0].message.content)

            except Exception as e:
                st.error(f"Error generating insight: {e}")
