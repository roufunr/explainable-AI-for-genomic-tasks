from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch, numpy as np
from lime.lime_text import LimeTextExplainer
from tqdm.auto import tqdm   
import matplotlib.pyplot as plt
import pandas as pd
import time

start_time = time.time()
 
ckpt   = "/home/rouf/data/NMT"
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda")
 
tokenizer = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
model     = (AutoModelForSequenceClassification
             .from_pretrained(ckpt, trust_remote_code=True)
             .to(device)
             .eval())

def predict_proba(bpe_strings, batch_size=256):
    """LIME → probs on GPU, with a tqdm bar."""
    probs = []
    for i in tqdm(range(0, len(bpe_strings), batch_size),
                  desc="LIME batches", leave=False):
        batch = bpe_strings[i:i+batch_size]
 
        enc = tokenizer(batch,
                        padding=True,
                        truncation=True,
                        add_special_tokens=True,   # CLS/SEP go back in
                        return_tensors="pt").to(device)
 
        with torch.no_grad():
            logits = model(**enc).logits
            probs.append(torch.softmax(logits, dim=-1).cpu())
    return torch.cat(probs).numpy()


raw_seq = ("CTAAATATTAACTGGTCTTGTGAGATGTCTTCTTGGCTGGAGCCTGACCACAGGAAGAAGAGGCGCCCGGAAAACCTTAGCTCTT")
 
# Tokenise once with DNABERT‑2’s BPE, then join with spaces
tokens = tokenizer.tokenize(raw_seq)
lime_string = " ".join(tokens)          # e.g. 'C TAA AT ATTA ACT GG TCTT ...'
 
explainer = LimeTextExplainer(
    split_expression=r"\s+",            # split on spaces
    bow=False,
    class_names=list(model.config.id2label.values())
)
 
exp = explainer.explain_instance(
    lime_string,
    predict_proba,
    num_features=15,
    num_samples=4000                    # ↑ for smoother importance curves
)

end_time = time.time()

print(exp.as_list())

# Get explanation as list
lime_results = exp.as_list()

# Convert to DataFrame
df = pd.DataFrame(lime_results, columns=["Token", "Importance"])

# Sort: positive first, then negative
df_sorted = pd.concat([
    df[df["Importance"] > 0].sort_values("Importance", ascending=False),
    df[df["Importance"] < 0].sort_values("Importance")
])
plt.figure(figsize=(8, 6))
colors = df_sorted["Importance"].apply(lambda x: "green" if x > 0 else "red")
plt.barh(df_sorted["Token"], df_sorted["Importance"], color=colors)
plt.title("LIME Explanation: Green First")
plt.xlabel("Importance")
plt.gca().invert_yaxis()  # So the top bar is the highest importance
plt.tight_layout()

# Save the figure
plt.savefig("lime_explanation_custom_sorted.png", dpi=300)
plt.close()

print("✅ Custom sorted LIME plot saved as 'lime_explanation_custom_sorted.png'")
print(end_time - start_time)