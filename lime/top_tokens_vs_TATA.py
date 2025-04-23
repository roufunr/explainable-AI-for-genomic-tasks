import os
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt

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

# Convert the dictionary to a DataFrame
token_df = pd.DataFrame(token_counts.items(), columns=['Token', 'FileCount'])

# Find the top TATA-containing token
tata_df = token_df[token_df["Token"].str.contains("TATA")]
top_tata_token = tata_df.iloc[0] if not tata_df.empty else None
print(top_tata_token)

# Get top 10 tokens by file count
top_10_df = token_df.sort_values(by='FileCount', ascending=False).head(10)
print(top_10_df)

# Plotting
tokens = top_10_df['Token'].tolist()
file_counts = top_10_df['FileCount'].tolist()

tata_token = top_tata_token['Token'] if top_tata_token is not None else "TCTATA"
tata_count = top_tata_token['FileCount'] if top_tata_token is not None else 0

plt.figure(figsize=(12, 6))
bars = plt.bar(tokens, file_counts, label='Top 10 Tokens')
plt.axhline(y=tata_count, color='red', linestyle='--', label=f'{tata_token} (Total Sequences={tata_count})')
plt.text(len(tokens)-1, tata_count+5, f'{tata_token}: {tata_count}', color='red', ha='right')

plt.title("Top 10 Most Frequent Tokens and TCTATA's Position")
plt.xlabel("Token")
plt.ylabel("File Count")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.grid(axis='y')
plt.savefig("top_tokens_plot.png", dpi=300)
