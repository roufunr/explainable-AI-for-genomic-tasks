import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

def find_best_tata_row(file_path):
    df = pd.read_csv(file_path)
    best_idx = None
    best_score = float('-inf')

    for idx, row in df.iterrows():
        tokens = row.get('anchor', '').split(" AND ")
        vals = row.get('importance_values', "")
        try:
            imps = [float(x) for x in vals.split(';')]
        except ValueError:
            print(f"Row {idx}: could not parse importance_values, skipping")
            continue

        # trim to shortest length
        n = min(len(tokens), len(imps))
        tokens, imps = tokens[:n], imps[:n]

        for tok, score in zip(tokens, imps):
            if "TATA" in tok and score > best_score and row.get('model_accuracy') == 'Correct':
                best_score, best_idx = score, idx

    if best_idx is None:
        print("No TATA tokens found in any row.")
        return None
    return df.loc[best_idx]

def visualize_row(row, save_path="tata_importance.png"):
    if row is None:
        return

    # Prepare tokens & importance
    tokens = row['anchor'].split(" AND ")
    imps = [float(x) for x in row['importance_values'].split(';')]
    n = min(len(tokens), len(imps))
    tokens, imps = tokens[:n], imps[:n]

    # Plot horizontal bars
    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = range(n)
    bars = ax.barh(y_pos, imps, color='skyblue')
    for i, tok in enumerate(tokens):
        if "TATA" in tok:
            bars[i].set_color('red')

    # Labels & ticks
    ax.set_yticks(y_pos)
    ax.set_yticklabels(tokens)
    ax.invert_yaxis()  # highest at top
    ax.set_xlabel('Importance Value')
    ax.set_ylabel('Token')
    
    title = f"Row {row.name}"
    if 'true_label' in row and 'predicted_label' in row:
        title += f" — True: {row['true_label']}, Pred: {row['predicted_label']}"
    ax.set_title(title)

    # Legend
    legend_elems = [
        Patch(facecolor='red', label='Contains TATA'),
        Patch(facecolor='skyblue', label='Other tokens')
    ]
    ax.legend(handles=legend_elems, loc='lower right')

    plt.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    # Print metrics & values
    for fld in ('model_accuracy','precision','coverage'):
        if fld in row:
            print(f"{fld}: {row[fld]}")
    print("\nTokens and importances:")
    for tok, score in zip(tokens, imps):
        mark = " (TATA)" if "TATA" in tok else ""
        print(f"  {tok}{mark}: {score}")


if __name__ == "__main__":
    best = find_best_tata_row("result.csv")
    visualize_row(best)
