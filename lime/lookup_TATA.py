import pandas as pd
import os


results = []
data_folder = "/home/rouf/data/raw/lime_exp_results/csv"
dataset = pd.read_csv("../dataset.csv")
# Loop through all files from 0.csv to 5928.csv
for i in range(5929):
    file_path = os.path.join(data_folder, f"{i}.csv")
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        filtered_df = df[df['Token'].str.contains("TATA")]
        for _, row in filtered_df.iterrows():
            results.append({
                "seq_id": i,
                "token": row['Token'],
                "importance": row['Importance'],
                "label": dataset.iloc[i]["label"],
                "sequence": dataset.iloc[i]["sequence"]
            })

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Save to CSV
output_csv_path = os.path.join(data_folder, "/home/rouf/data/raw/lime_exp_results/filtered_TATA_sequences.csv")
results_df.to_csv(output_csv_path, index=False)
