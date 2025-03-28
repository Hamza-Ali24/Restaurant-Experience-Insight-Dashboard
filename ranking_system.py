import pandas as pd
import os

# Load the dataset
file_path = "mot_&_sentiment_analysis/restaurant_mot_&_sentiment_analysis_batch_results.csv"
df = pd.read_csv(file_path)

# Target businesses
target_restaurants = ["Parrot's Cay Tavern & Grill", "Mezzaluna", "The Starving Rooster"]

# MoT categories
MOT_CATEGORIES = [
    "Arrival & First Impressions", "Waiting Time", "Ambience & Atmosphere",
    "Service Interaction", "Menu Presentation & Ordering", "Food & Drink Arrival Time",
    "Food Quality & Presentation", "Handling of Dietary Requirements", "Toilet Cleanliness & Maintenance",
    "Billing & Payment Process", "Issue Resolution & Complaint Handling", "Word-of-Mouth & Recommendations"
]

MOT_SENTIMENT_CATEGORIES = [f"{mot}_Sentiment" for mot in MOT_CATEGORIES]

for restaurant in target_restaurants:
    print(f"\n Ranking for: {restaurant}")

    df_restaurant = df[df["name_y"] == restaurant]

    # ----- Table 1: Importance Ranking (Descending by Mentions) -----
    mot_mentions = df_restaurant[MOT_CATEGORIES].sum().reset_index()
    mot_mentions.columns = ["MOT", "Mentions"]
    mot_mentions["Avg_Sentiment"] = ""
    mot_mentions["Type"] = "Importance"
    mot_mentions = mot_mentions.sort_values(by="Mentions", ascending=False)

    # ----- Table 2: Satisfaction Ranking (Ascending by Avg Sentiment) -----
    sentiment_data = []
    for mot_sentiment in MOT_SENTIMENT_CATEGORIES:
        mot_name = mot_sentiment.replace("_Sentiment", "")
        valid_values = df_restaurant[df_restaurant[mot_sentiment] > 0][mot_sentiment]
        avg = round(valid_values.mean(), 2) if not valid_values.empty else None
        sentiment_data.append({"MOT": mot_name, "Mentions": "", "Avg_Sentiment": avg, "Type": "Satisfaction"})

    sentiment_df = pd.DataFrame(sentiment_data)
    sentiment_df = sentiment_df.sort_values(by="Avg_Sentiment", ascending=True)

    # Combine both tables
    combined = pd.concat([mot_mentions, sentiment_df], ignore_index=True)
    combined = combined[["Type", "MOT", "Mentions", "Avg_Sentiment"]]

    # Save to CSV
    output_filename = f"{restaurant.replace(' ', '_').replace('&', 'and')}_mot_&_sentiment_ranking.csv"
    output_path = os.path.join("ranking", output_filename)
    combined.to_csv(output_path, index=False, float_format="%.2f")

    print(f"Rankings saved to: {output_path}")

print("\n All rankings generated and saved!")
