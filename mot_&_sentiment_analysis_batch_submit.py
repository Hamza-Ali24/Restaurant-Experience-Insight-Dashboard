import openai
import pandas as pd
import json
import os               # To access environment variables
from dotenv import load_dotenv   # To load variables from your .env file
load_dotenv()

# OpenAI API Key
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found. Make sure it's set in your .env file!")
# Initialise OpenAI Client
client = openai.OpenAI(api_key=openai_api_key)

# File paths
input_csv_path = "dataset/cleaned_reviews.csv"
jsonl_output_path = "batch_input.jsonl"
model = "gpt-4o-mini"

# Load cleaned reviews
df = pd.read_csv(input_csv_path)

# Define Moments of Truth (MoTs)
MOT_CATEGORIES = [
    "Arrival & First Impressions", "Waiting Time", "Ambience & Atmosphere",
    "Service Interaction", "Menu Presentation & Ordering", "Food & Drink Arrival Time",
    "Food Quality & Presentation", "Handling of Dietary Requirements", "Toilet Cleanliness & Maintenance",
    "Billing & Payment Process", "Issue Resolution & Complaint Handling", "Word-of-Mouth & Recommendations"
]

# Generate prompt for each review
def generate_prompt(review_text):
    return f"""
You are an AI expert in restaurant review analysis.

Your task is to analyse the provided customer review:
1️. Identify whether each MoT is mentioned in the review.
2️. If a review does not explicitly mention an MoT, infer reasonable ones based on context.
3️. For each MoT, mark it as 1 (mentioned) or 0 (not mentioned).
4️. For each MoT, determine the sentiment as:
    - 3 for "Positive"
    - 2 for "Neutral"
    - 1 for "Negative"
    - 0 for "Not Mentioned"

Return your results in strict JSON format. No extra text.

MoT Categories:
{", ".join(MOT_CATEGORIES)}

Review to analyse:
\"{review_text}\"

JSON Response Format Example:
{{
    "Arrival & First Impressions": 1,
    "Arrival & First Impressions_Sentiment": 2,
    "Waiting Time": 0,
    "Waiting Time_Sentiment": 1,
    ...
}}
"""

# Create the batch input JSONL
batch_inputs = []

for idx, row in df.iterrows():
    review_text = row["text"]
    prompt = generate_prompt(review_text)

    request_payload = {
        "custom_id": f"review-{idx}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are an AI expert in restaurant review analysis."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0
        }
    }

    batch_inputs.append(request_payload)

# Save batch requests to JSONL
with open(jsonl_output_path, "w", encoding="utf-8") as outfile:
    for entry in batch_inputs:
        json.dump(entry, outfile)
        outfile.write("\n")

print(f"JSONL file created at {jsonl_output_path}")

# Upload JSONL file to OpenAI
upload_response = client.files.create(
    file=open(jsonl_output_path, "rb"),
    purpose="batch"
)

batch_input_file_id = upload_response.id
print(f"Uploaded file. File ID: {batch_input_file_id}")

# Create the batch job
batch_response = client.batches.create(
    input_file_id=batch_input_file_id,
    endpoint="/v1/chat/completions",
    completion_window="24h",
    metadata={"description": "Restaurant MoT Analysis Batch"}
)

batch_id = batch_response.id
print(f"Batch job created! Batch ID: {batch_id}")
print("You can track progress on the OpenAI dashboard or by using the batch ID.")
