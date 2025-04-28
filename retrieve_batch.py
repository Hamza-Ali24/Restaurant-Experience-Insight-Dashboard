import openai
import pandas as pd
import openai
import json
import os               # To access environment variables
from dotenv import load_dotenv   # To load variables from your .env file
load_dotenv()

# open api key
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found. Make sure it's set in your .env file!")

batch_id = "batch_67dcf90b1ce48190b69b653677a3be12"  # Replace with the actual batch ID
output_csv_path = "mot_&_sentiment_analysis/restaurant_mot_&_sentiment_analysis_batch_results.csv"

# Original cleaned csv for merging
cleaned_reviews_csv_path = "dataset/cleaned_reviews.csv"

#Initialise OpenAI Client
client = openai.OpenAI(api_key=openai_api_key)

#Get batch status
batch = client.batches.retrieve(batch_id)
print(f" Batch Status: {batch.status}")

if batch.status != "completed":
    print("Batch is not yet completed. Try again later.")
    exit()

#Download output file
output_file_id = batch.output_file_id
output_file_content = client.files.content(output_file_id)
output_lines = output_file_content.text.strip().split("\n")

#Parse output JSONL
results = []
for line in output_lines:
    item = json.loads(line)
    result = {
        "custom_id": item["custom_id"],
        "response": item["response"]["body"]["choices"][0]["message"]["content"]
    }
    results.append(result)

#Define MoTs for validation
MOT_CATEGORIES = [
    "Arrival & First Impressions", "Waiting Time", "Ambience & Atmosphere",
    "Service Interaction", "Menu Presentation & Ordering", "Food & Drink Arrival Time",
    "Food Quality & Presentation", "Handling of Dietary Requirements", "Toilet Cleanliness & Maintenance",
    "Billing & Payment Process", "Issue Resolution & Complaint Handling", "Word-of-Mouth & Recommendations"
]
# Define sentiment values
valid_sentiments = {0, 1, 2, 3}

#Process JSON responses & validate sentiment
rows = []
for res in results:
    content = res["response"]

    try:
        json_data = json.loads(content)
    except json.JSONDecodeError:
        print(f" Failed to parse JSON for {res['custom_id']}")
        continue

    # Validate sentiment values
    for mot in MOT_CATEGORIES:
        sentiment_key = f"{mot}_Sentiment"
        sentiment_value = json_data.get(sentiment_key)

        if sentiment_value not in valid_sentiments:
            print(
                f" Invalid sentiment value '{sentiment_value}' for '{sentiment_key}' in {res['custom_id']}. Defaulting to 0.")
            json_data[sentiment_key] = 0

    json_data["custom_id"] = res["custom_id"]
    rows.append(json_data)

# Convert to dataframe
df_results = pd.DataFrame(rows)

#Merge with cleaned reviews
df_cleaned = pd.read_csv(cleaned_reviews_csv_path)

# Merge on custom_id
# Extract row index from custom_id
df_results["review_index"] = df_results["custom_id"].str.extract(r'review-(\d+)').astype(int)

df_merged = df_cleaned.merge(df_results, left_index=True, right_on="review_index", how="inner")

# Drop helper columns
df_merged.drop(columns=["custom_id", "review_index"], inplace=True)

#Save final csv
df_merged.to_csv(output_csv_path, index=False)
print(f" Results saved to: {output_csv_path}")
