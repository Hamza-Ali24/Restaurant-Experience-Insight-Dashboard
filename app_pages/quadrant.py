import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
import openai
from dotenv import load_dotenv

def show_quadrant(selected_businesses):
    st.title("📊 MOT Priority Matrix")

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

    # Prepare data
    imp = df[df["Type"] == "Importance"].copy()
    sat = df[df["Type"] == "Satisfaction"].copy()
    merged = pd.merge(imp, sat, on="MOT", suffixes=("_Importance", "_Satisfaction"))
    merged["Mentions"] = pd.to_numeric(merged["Mentions_Importance"], errors="coerce")
    merged["Avg_Sentiment"] = pd.to_numeric(merged["Avg_Sentiment_Satisfaction"], errors="coerce")
    merged = merged.dropna(subset=["Mentions", "Avg_Sentiment"])

    # Quadrant splits
    x_split = merged["Avg_Sentiment"].median()
    y_split = merged["Mentions"].median()

    def quadrant_label(row):
        if row["Avg_Sentiment"] < x_split and row["Mentions"] > y_split:
            return "🚨 Fix Now"
        elif row["Avg_Sentiment"] >= x_split and row["Mentions"] > y_split:
            return "🌟 Invest Further"
        elif row["Avg_Sentiment"] < x_split and row["Mentions"] <= y_split:
            return "🤷 Deprioritise"
        else:
            return "👍 Maintain"

    merged["Quadrant"] = merged.apply(quadrant_label, axis=1)

    quadrant_colors = {
        "🚨 Fix Now": "red",
        "🌟 Invest Further": "green",
        "🤷 Deprioritise": "orange",
        "👍 Maintain": "blue"
    }

    x_min, x_max = merged["Avg_Sentiment"].min(), merged["Avg_Sentiment"].max()
    y_min, y_max = merged["Mentions"].min(), merged["Mentions"].max()

    fig = go.Figure()

    # Shaded quadrants
    fig.add_shape(type="rect", x0=x_min, x1=x_split, y0=y_split, y1=y_max,
                  fillcolor="rgba(255, 100, 100, 0.1)", line_width=0, layer="below")
    fig.add_shape(type="rect", x0=x_split, x1=x_max, y0=y_split, y1=y_max,
                  fillcolor="rgba(100, 255, 100, 0.1)", line_width=0, layer="below")
    fig.add_shape(type="rect", x0=x_min, x1=x_split, y0=y_min, y1=y_split,
                  fillcolor="rgba(255, 200, 0, 0.1)", line_width=0, layer="below")
    fig.add_shape(type="rect", x0=x_split, x1=x_max, y0=y_min, y1=y_split,
                  fillcolor="rgba(100, 100, 255, 0.1)", line_width=0, layer="below")

    # Data points
    for quadrant, group in merged.groupby("Quadrant"):
        fig.add_trace(go.Scatter(
            x=group["Avg_Sentiment"],
            y=group["Mentions"],
            mode="markers",
            marker=dict(size=14, color=quadrant_colors[quadrant]),
            name=quadrant,
            hovertemplate="<b>%{text}</b><br>Satisfaction: %{x}<br>Mentions: %{y}<extra></extra>",
            text=group["MOT"]
        ))

    # Median lines
    fig.add_shape(type="line", x0=x_split, x1=x_split, y0=y_min, y1=y_max,
                  line=dict(dash="dash", color="grey"))
    fig.add_shape(type="line", x0=x_min, x1=x_max, y0=y_split, y1=y_split,
                  line=dict(dash="dash", color="grey"))

    # Quadrant labels
    fig.add_annotation(x=x_min, y=y_max, text="🚨 Fix Now", showarrow=False, xanchor="left", yanchor="top", font=dict(size=13))
    fig.add_annotation(x=x_max, y=y_max, text="🌟 Invest Further", showarrow=False, xanchor="right", yanchor="top", font=dict(size=13))
    fig.add_annotation(x=x_min, y=y_min, text="🤷 Deprioritise", showarrow=False, xanchor="left", yanchor="bottom", font=dict(size=13))
    fig.add_annotation(x=x_max, y=y_min, text="👍 Maintain", showarrow=False, xanchor="right", yanchor="bottom", font=dict(size=13))

    fig.update_layout(
        title=f"{selected} - MoT Priority Matrix",
        xaxis_title="Customer Satisfaction (Avg Sentiment)",
        yaxis_title="Importance (Mentions)",
        legend_title="Quadrant",
        height=600,
        template="plotly"
    )

    st.plotly_chart(fig, use_container_width=True)

    # 🎯 How to interpret the quadrants
    with st.expander("🎯 How to Interpret the Quadrants"):
        st.markdown("""
**Quadrant 1 (🚨 Fix Now):**  
➡️ High Importance, Low Satisfaction  
Prioritise fixing these areas immediately.  

**Quadrant 2 (🌟 Invest Further):**  
➡️ High Importance, High Satisfaction  
Maintain and enhance these strengths.  

**Quadrant 3 (🤷 Deprioritise):**  
➡️ Low Importance, Low Satisfaction  
Lower priority for immediate action.  

**Quadrant 4 (👍 Maintain):**  
➡️ Low Importance, High Satisfaction  
Keep these areas stable and monitored.
""")

    # GPT-4 Insight Generator
    st.subheader("💡 Generate Insight from Matrix")

    if st.button("Generate Insight") and client:
        with st.spinner("Analysing quadrant matrix with GPT-4..."):
            try:
                summaries = []
                for group in ["🚨 Fix Now", "🌟 Invest Further", "🤷 Deprioritise", "👍 Maintain"]:
                    sub = merged[merged["Quadrant"] == group]
                    if not sub.empty:
                        points = "; ".join([
                            f"{row['MOT']} (Mentions: {int(row['Mentions'])}, Sentiment: {row['Avg_Sentiment']:.2f})"
                            for _, row in sub.iterrows()
                        ])
                        summaries.append(f"**{group}**:\n{points}\n")
                    else:
                        summaries.append(f"**{group}**:\n_No items in this quadrant._\n")

                chart_summary = "\n".join(summaries)

                prompt = (
                    f"You are analysing customer feedback for the business '{selected}'.\n\n"
                    f"Here is the distribution of Moments of Truth (MoTs) based on Importance and Satisfaction:\n\n"
                    f"{chart_summary}\n\n"
                    "Please explain what needs urgent attention (🚨), what should be enhanced (🌟), "
                    "what can be deprioritised (🤷), and what should be maintained (👍)."
                )

                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a strategic customer experience consultant."},
                        {"role": "user", "content": prompt}
                    ]
                )
                st.success(response.choices[0].message.content)

            except Exception as e:
                st.error(f"Error generating insight: {e}")
