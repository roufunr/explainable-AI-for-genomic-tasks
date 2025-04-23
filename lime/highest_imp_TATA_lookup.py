import os
import pandas as pd

# Define the directory path
base_dir = "/home/rouf/data/raw/lime_exp_results/csv"

# Total number of CSVs expected
num_files = 5929  # 0 to 5928

# Track the max importance and corresponding file/token
max_importance = -float('inf')
max_info = {"file": None, "token": None, "importance": None}

# Iterate through each file
for i in range(num_files):
    file_path = os.path.join(base_dir, f"{i}.csv")
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path)
            df_tata = df[df['Token'].str.contains("TATA", na=False)]
            if not df_tata.empty:
                max_row = df_tata.loc[df_tata['Importance'].idxmax()]
                if max_row['Importance'] > max_importance:
                    max_importance = max_row['Importance']
                    max_info = {
                        "file": f"{i}.csv",
                        "token": max_row['Token'],
                        "importance": max_row['Importance']
                    }
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")

print(max_info)