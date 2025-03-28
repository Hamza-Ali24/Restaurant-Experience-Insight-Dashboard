import pandas as pd
import json

#Load full reviews dataset
reviews_path = "dataset/review-North_Dakota.json"

with open(reviews_path, "r", encoding="utf-8") as f:
    reviews_data = [json.loads(line) for line in f]

# Convert reviews data into a DataFrame
reviews_df = pd.DataFrame(reviews_data)

#Load Business Metadata
metadata_path = "dataset/meta-North_Dakota.json"

with open(metadata_path, "r", encoding="utf-8") as f:
    meta_data = [json.loads(line) for line in f]

# Convert metadata into a DataFrame
meta_df = pd.DataFrame(meta_data)

#Filter for Selected Businesses
selected_gmap_ids = [
    "0x52c8c966d72c4cab:0x7781486659278013",  # Business 1
    "0x52c680ca5f0013f9:0x5940d092e5ebda66",  # Business 2
    "0x52df2a1bbfbef843:0x9dc457e28736f954"  # Business 3
]

# Filter only the reviews for the selected businesses
filtered_reviews_df = reviews_df[reviews_df["gmap_id"].isin(selected_gmap_ids)]

#Merge Reviews with Business Metadata
merged_df = filtered_reviews_df.merge(meta_df, on="gmap_id", how="left")

#Remove Reviews with Missing Text
cleaned_reviews_df = merged_df.dropna(subset=["text"])

#Save Cleaned Processed Data
output_path = "dataset/cleaned_reviews.csv"
cleaned_reviews_df.to_csv(output_path, index=False)

print("Processing Complete! Cleaned review data saved at:")
print(f"- {output_path}")
