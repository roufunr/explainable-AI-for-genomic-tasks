# DNA Sequence Analysis Script
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from collections import Counter

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk")

# Load data
df = pd.read_csv('result.csv')

# Create TATA features
df['contains_tata'] = df['anchor'].apply(lambda x: 'TATA' in x if isinstance(x, str) else False)

# Parse importance values
def parse_importance(imp_str):
    if isinstance(imp_str, str):
        return [float(x) for x in imp_str.strip('"').split(';')]
    return []

# Parse anchors
def parse_anchors(anchor_str):
    if isinstance(anchor_str, str):
        return anchor_str.strip('"').split(' AND ')
    return []

# Apply parsing
df['importance_list'] = df['importance_values'].apply(parse_importance)
df['anchors_list'] = df['anchor'].apply(parse_anchors)
df['avg_importance'] = df['importance_list'].apply(lambda x: np.mean(x) if len(x) > 0 else 0)

# Create output directory
if not os.path.exists('output'):
    os.makedirs('output')

# 1. Importance visualization
plt.figure(figsize=(12, 8))
sns.boxplot(x='true_label', y='avg_importance', data=df)
plt.title('Importance by Label')
plt.savefig('output/importance_by_label.png', dpi=300)
plt.close()

plt.figure(figsize=(12, 8))
sns.boxplot(x='contains_tata', y='avg_importance', hue='true_label', data=df)
plt.title('Importance: TATA vs Non-TATA')
plt.savefig('output/tata_importance.png', dpi=300)
plt.close()

# 2. TATA analysis
tata_counts = df.groupby(['true_label', 'contains_tata']).size().reset_index(name='count')
tata_stats = pd.pivot_table(df, values='avg_importance', 
                           index='true_label', columns='contains_tata',
                           aggfunc=['mean', 'count', 'std'])
tata_stats.to_csv('output/tata_stats.csv')

plt.figure(figsize=(10, 6))
sns.barplot(x='true_label', y='count', hue='contains_tata', data=tata_counts)
plt.title('TATA Occurrence by Label')
plt.savefig('output/tata_counts.png', dpi=300)
plt.close()

# 3. TATA importance by quadrant
tata_quadrants = {
    'Label 0, Negative': ((df['true_label'] == 0) & (df['avg_importance'] < 0) & df['contains_tata']).sum(),
    'Label 0, Positive': ((df['true_label'] == 0) & (df['avg_importance'] >= 0) & df['contains_tata']).sum(),
    'Label 1, Negative': ((df['true_label'] == 1) & (df['avg_importance'] < 0) & df['contains_tata']).sum(),
    'Label 1, Positive': ((df['true_label'] == 1) & (df['avg_importance'] >= 0) & df['contains_tata']).sum()
}

quad_df = pd.DataFrame({'Count': tata_quadrants}).reset_index().rename(columns={'index': 'Category'})
quad_df.to_csv('output/tata_quadrants.csv', index=False)

plt.figure(figsize=(10, 6))
plt.bar(quad_df['Category'], quad_df['Count'])
plt.title('TATA Sequence Analysis by Label & Importance')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('output/tata_quadrants.png', dpi=300)
plt.close()

# 4. Coverage vs Precision
plt.figure(figsize=(10, 8))
sns.scatterplot(data=df, x='coverage', y='precision', hue='contains_tata', style='true_label')
plt.title('Coverage vs Precision')
plt.savefig('output/coverage_precision.png', dpi=300)
plt.close()

# 5. Nucleotide analysis
def count_nucleotides(sequences):
    counts = {'A': 0, 'C': 0, 'G': 0, 'T': 0}
    total = 0
    for seq in sequences:
        for nucleotide in seq:
            if nucleotide in counts:
                counts[nucleotide] += 1
                total += 1
    return {n: (count / total) * 100 if total > 0 else 0 for n, count in counts.items()}

# Extract sequences by label
label_0_seqs = [kmer for idx, anchors in enumerate(df['anchors_list']) 
               for kmer in anchors if df.iloc[idx]['true_label'] == 0]
label_1_seqs = [kmer for idx, anchors in enumerate(df['anchors_list']) 
               for kmer in anchors if df.iloc[idx]['true_label'] == 1]

# Count nucleotides
nuc_0 = count_nucleotides(label_0_seqs)
nuc_1 = count_nucleotides(label_1_seqs)

# Save nucleotide composition
pd.DataFrame({
    'Label 0 (%)': nuc_0,
    'Label 1 (%)': nuc_1
}).to_csv('output/nucleotide_composition.csv')

# Plot nucleotide composition
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.bar(nuc_0.keys(), nuc_0.values(), color='salmon')
plt.title('Nucleotides - Label 0')
plt.subplot(1, 2, 2)
plt.bar(nuc_1.keys(), nuc_1.values(), color='skyblue')
plt.title('Nucleotides - Label 1')
plt.tight_layout()
plt.savefig('output/nucleotide_composition.png', dpi=300)
plt.close()

# 6. TATA position analysis
tata_positions = []
for idx, row in df.iterrows():
    if row['contains_tata']:
        for anchor in row['anchors_list']:
            pos = anchor.find('TATA')
            if pos >= 0:
                tata_positions.append(pos)

if tata_positions:
    plt.figure(figsize=(10, 6))
    plt.hist(tata_positions, bins=range(0, max(tata_positions)+2), color='green')
    plt.title('TATA Box Position Distribution')
    plt.xlabel('Position')
    plt.ylabel('Count')
    plt.savefig('output/tata_positions.png', dpi=300)
    plt.close()

# 7. Summary statistics
summary = {
    'Total Sequences': len(df),
    'Label 0 Count': (df['true_label'] == 0).sum(),
    'Label 1 Count': (df['true_label'] == 1).sum(),
    'Contains TATA': df['contains_tata'].sum(),
    'Label 0 with TATA': ((df['true_label'] == 0) & df['contains_tata']).sum(),
    'Label 1 with TATA': ((df['true_label'] == 1) & df['contains_tata']).sum(),
    'TATA in Label 0, Negative': ((df['true_label'] == 0) & (df['avg_importance'] < 0) & df['contains_tata']).sum(),
    'TATA in Label 0, Positive': ((df['true_label'] == 0) & (df['avg_importance'] >= 0) & df['contains_tata']).sum(),
    'TATA in Label 1, Negative': ((df['true_label'] == 1) & (df['avg_importance'] < 0) & df['contains_tata']).sum(),
    'TATA in Label 1, Positive': ((df['true_label'] == 1) & (df['avg_importance'] >= 0) & df['contains_tata']).sum()
}

pd.DataFrame({'Value': summary}).to_csv('output/summary.csv')

print("Analysis complete! All charts and tables saved to 'output' folder.")
