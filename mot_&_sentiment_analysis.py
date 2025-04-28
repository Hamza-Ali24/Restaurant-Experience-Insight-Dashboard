import openai
import pandas as pd
import json
import time
import os               # To access environment variables
from dotenv import load_dotenv   # To load variables from your .env file
load_dotenv()
# OpenAI API Key
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError(" OPENAI_API_KEY not found. Make sure it's set in your .env file!")
# Initialise OpenAI Client
client = openai.OpenAI(api_key=openai_api_key)

# Load dataset
file_path = "dataset/cleaned_reviews.csv"
df = pd.read_csv(file_path)

# Filter dataset for the three specific restaurants
target_restaurants = ["Parrot's Cay Tavern & Grill", "Mezzaluna", "The Starving Rooster"]
df_filtered = df[df["name_y"].isin(target_restaurants)].copy()

# Define Moments of Truth (MoTs)
MOT_CATEGORIES = [
    "Arrival & First Impressions", "Waiting Time", "Ambience & Atmosphere",
    "Service Interaction", "Menu Presentation & Ordering", "Food & Drink Arrival Time",
    "Food Quality & Presentation", "Handling of Dietary Requirements", "Toilet Cleanliness & Maintenance",
    "Billing & Payment Process", "Issue Resolution & Complaint Handling", "Word-of-Mouth & Recommendations"
]

# JSON Schema for validation
json_schema = {
    f"{mot}": {"type": "boolean"} for mot in MOT_CATEGORIES
}
json_schema.update({
    f"{mot}_Sentiment": {
        "type": "integer",
        "enum": [0, 1, 2, 3]  # 0 = Not Mentioned, 1 = Negative, 2 = Neutral, 3 = Positive
    } for mot in MOT_CATEGORIES
})

# Required Properties
required_properties = [
    *MOT_CATEGORIES,
    *[f"{mot}_Sentiment" for mot in MOT_CATEGORIES]
]


#ChatGPT API prompt
def generate_prompt(review_text):
    return f"""
You are an AI expert in **restaurant review analysis**.

**You will be given a review and your task is:**
1️. Identify **all relevant Moments of Truth (MoTs)** related to the dining experience.
2️. If a review does not explicitly mention an MoT, infer reasonable ones based on context.
3️. For each MoT, **mark it as 1 (mentioned) or 0 (not mentioned)**.
4️. For each MoT, determine the **sentiment** as:
    - 3 for "Positive"
    - 2 for "Neutral"
    - 1 for "Negative"
    - 0 for "Not Mentioned"

5️. Return your results in **strict JSON format**. No extra text.

**MoT Categories:**
{", ".join(MOT_CATEGORIES)}

**Review to analyse:**
\"{review_text}\"

**JSON Response Format Example:**
{{
    "Arrival & First Impressions": 1,
    "Arrival & First Impressions_Sentiment": 3,
    "Waiting Time": 0,
    "Waiting Time_Sentiment": 0,
    ...
}}
"""


# Analyse review function
def analyse_review_with_chatgpt(review_text, max_retries=3):
    if not isinstance(review_text, str) or review_text.strip() == "":
        print("Skipping empty review text.")
        return {**{f"{mot}": 0 for mot in MOT_CATEGORIES},
                **{f"{mot}_Sentiment": 0 for mot in MOT_CATEGORIES}}

    prompt = generate_prompt(review_text)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an AI expert in restaurant review analysis."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )

            response_text = response.choices[0].message.content

            # Debugging: Print raw ChatGPT response
            print("\nRaw ChatGPT Response:\n", response_text)

            # Attempt to parse JSON output from ChatGPT
            response_data = json.loads(response_text)

            # Ensure all MOTs and Sentiments are present
            for mot in MOT_CATEGORIES:
                response_data.setdefault(mot, 0)
                response_data.setdefault(f"{mot}_Sentiment", 0)  # fallback to "Not Mentioned"

            return response_data

        except json.JSONDecodeError:
            print(f"Error: Invalid JSON from ChatGPT (Attempt {attempt + 1}/{max_retries}). Retrying...")
            time.sleep(2)

    print("ChatGPT failed to return valid JSON after multiple attempts. Skipping review.")
    return {**{f"{mot}": 0 for mot in MOT_CATEGORIES},
            **{f"{mot}_Sentiment": 0 for mot in MOT_CATEGORIES}}


# Limit to 20 reviews for testing
df_filtered_sample = df_filtered.sample(n=20, random_state=42)

# Apply the analysis function
df_filtered_sample["ChatGPT Analysis"] = df_filtered_sample["text"].apply(analyse_review_with_chatgpt)

# Extract columns for MOT flags and Sentiment scores
for mot in MOT_CATEGORIES:
    df_filtered_sample[mot] = df_filtered_sample["ChatGPT Analysis"].apply(lambda x: x.get(mot, 0))
    df_filtered_sample[f"{mot}_Sentiment"] = df_filtered_sample["ChatGPT Analysis"].apply(
        lambda x: x.get(f"{mot}_Sentiment", 0))

# Drop the raw JSON column before saving
df_filtered_sample.drop(columns=["ChatGPT Analysis"], inplace=True)

# Save the processed data into CSV
df_filtered_sample.to_csv("mot_&_sentiment_analysis/restaurant_mot_&_sentiment_analysis.csv", index=False)

# Print completion message
print("Analysis complete! Results saved as 'restaurant_mot_&_sentiment_analysis.csv'.")
