import matplotlib.pyplot as plt

# Data
categories = ['Total', 'Avg Positive', 'Avg Negative']
sequence_counts = [5930, 3662, 2268]
tata_counts = [828, 763, 65]

# Plotting
plt.figure(figsize=(8, 6))
bar_width = 0.35
x = range(len(categories))

# Custom colors
sequence_colors = ['skyblue', 'skyblue', 'skyblue']
tata_colors = ['lightgreen', 'lightgreen', 'lightgreen']

# Bars
for i in range(len(categories)):
    plt.bar(x[i], sequence_counts[i], width=bar_width, color=sequence_colors[i], label='All Sequences' if i == 0 else "")
    plt.bar(x[i] + bar_width, tata_counts[i], width=bar_width, color=tata_colors[i], label='TATA tokens' if i == 0 else "")

# Labels and formatting
plt.xlabel('Token Categories')
plt.ylabel('Count')
plt.title('Token Distribution Overview')
plt.xticks([i + bar_width / 2 for i in x], categories)
plt.legend()
plt.tight_layout()

# Show plot
plt.show()
