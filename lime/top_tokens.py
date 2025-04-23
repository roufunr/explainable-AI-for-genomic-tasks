import os
import pandas as pd
from collections import defaultdict

# Define the directory path
base_dir = "/home/rouf/data/raw/lime_exp_results/csv"

# Dictionary to count token appearances across files
token_counts = defaultdict(int)

# Total number of CSVs expected
num_files = 5929  # 0 to 5928

# Iterate through each file
for i in range(num_files):
    file_path = os.path.join(base_dir, f"{i}.csv")
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path)
            unique_tokens = set(df['Token'].dropna())
            for token in unique_tokens:
                token_counts[token] += 1
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")

# Convert the dictionary to a DataFrame and sort
token_df = pd.DataFrame(token_counts.items(), columns=['Token', 'FileCount'])

tata_df = token_df[token_df["Token"].str.contains("TATA")]
top_tata_token = tata_df.iloc[0] if not tata_df.empty else None

print(top_tata_token)

token_df = token_df.sort_values(by='FileCount', ascending=False).head(10)
print(token_df)



tata_df = token_df[token_df["Token"].str.contains("TATA")]
top_tata_token = tata_df.iloc[0] if not tata_df.empty else None



