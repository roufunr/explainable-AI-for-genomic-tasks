import pandas as pd

# Replace this with your actual CSV file path
csv_path = "filtered_TATA_sequences.csv"

# Load the CSV
df = pd.read_csv(csv_path)

# Define the four conditions
cond_1 = ((df['importance'] > 0) & (df['label'] == 1))  # positive importance, label 1
cond_2 = ((df['importance'] < 0) & (df['label'] == 1))  # negative importance, label 1
cond_3 = ((df['importance'] > 0) & (df['label'] == 0))  # positive importance, label 0
cond_4 = ((df['importance'] < 0) & (df['label'] == 0))  # negative importance, label 0

# Count the occurrences
count_1 = cond_1.sum()
count_2 = cond_2.sum()
count_3 = cond_3.sum()
count_4 = cond_4.sum()

# Print results
print("Positive importance & label 1:", count_1)
print("Negative importance & label 1:", count_2)
print("Positive importance & label 0:", count_3)
print("Negative importance & label 0:", count_4)
