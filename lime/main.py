import os
import logging
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch, numpy as np
from lime.lime_text import LimeTextExplainer
from tqdm.auto import tqdm   
import matplotlib.pyplot as plt
import pandas as pd
import time

# ------------------ SETUP LOGGER ------------------
logging.basicConfig(
    filename='lime_run.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger()

ckpt = "/home/ab823254/data/NMT"
device = torch.device("cuda")

tokenizer = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
model = (AutoModelForSequenceClassification
         .from_pretrained(ckpt, trust_remote_code=True)
         .to(device)
         .eval())

def predict_proba(bpe_strings, batch_size=64):
    """LIME → probs on GPU, with a tqdm bar."""
    probs = []
    for i in tqdm(range(0, len(bpe_strings), batch_size), desc="LIME batches", leave=False):
        batch = bpe_strings[i:i+batch_size]

        enc = tokenizer(batch,
                        padding=True,
                        truncation=True,
                        add_special_tokens=True,
                        return_tensors="pt").to(device)

        with torch.no_grad():
            logits = model(**enc).logits
            probs.append(torch.softmax(logits, dim=-1).cpu())
    return torch.cat(probs).numpy()

raw_seq = ("CTGAAAGCTGAATACAGAAGGCATAGATGCTGCATCTTGAGTGTCCAGCTCTTCTGTGCTGGACATCAGGATGTATACACTTAACCTGGGAGGCTTAGAGTTGGGAAGAGAAGAGTTAGATCGAGGGCAAGGCTTCTGACTCCTCTCTGGGGAAAGAAGAATGGATACGTGTAGAGGTGCTATGTGCAAAAACAGCACCTCCACCTCTGCTACCCTCCCTGCAACACAACACACACACACACACACACACACGTACACACACACTCACTAAGGTCCTGCAAAGCCCGTGAAGAGACATACCACCTCTCCTTGCCAAAGGGTGTTCGGCCTTAAAATGCCATAAACAAAAACATATACAGAAATATTTTCGCTACAAGGACCCAGATGTCTAAGCAGTTTAACAGAAAGCTTGGCACCCCTGGGTGGTCATGCATGCCAACCAGATTTCTCATGCTTTAAGAAGTCTATTTTTTAAAACAGGATTGACCAAACTAATGGGGTCTGCTCTGGATGGTGAAAATGTTTACTTTCATTTCCCCCCAACCCCCTCGCCTCCCCAGTTCCCCCTGAATATATTCTCAAGAAAGAGTATGTCTTTGGGTGGTTGCAAGTAGGAAGTTTCAAGGCTTTTCTCCCTAACGAATACCGAACACTGAGCACACAGGAGGGGAACAGAAACTTGGAAGCAGAGCAAAGGACCAAGTTATCCTCCTGTCTAACCCTGCTCAACAGGATCACCCTGAAGTGTGGCTCTGAAGATATTTGTGAACTCTGGCACAGCTGTTTGGCAGAATTAATACTAAAACCACAGGTCTTGCAAAAACAAAGAATCAAAAAAAAAAAAAAAAAGTGAAAACCAAACAGCAGGAGGAAACCCTTAGGAAAGCTCGGTGTCTTTCAGACATGAGATTTTCTTTCTCCCACTCTACTTGACTGGCCTGACCTGGGGTCCAGTCCCACTTCCTGGTGGCTGCGGAAATCCGCAGTGAGGCTCAGTGTGGATTTTGGCTTCAGAGGATTTGGGAAATTCCACCTTTCAATCTGAAATCTGGGGAAAGTCCCAGAAAACAACGGATTACTATATTTCTCTCATGAACCAGACTGAAGCAAGGAGTCAGGTCTAGGACCAAGGACCTCGGCACTCCAGGGAGCCCAGGGCCCCTCAGAAAGGTCTCGCTTGGCCACCGAGAGCGTGGGGCCGTAGCAGTTGGCAAGCGGGGGAGGGGGGTCACGCACAACTGCTCTGGAGCCACACTGCATAAATGGCATTCGTTTTTGTTTTCTGATTAGAAAAACGTTGAAAAGAATACCAAAGTACAGAGAAAACACACCCCCATCTCCAATGTTTCCTCATTTATGTAAATATTCTCATATTTTATAAAACAGGGAGGTGCTGTCTATAATTTTTGCAGGCTTTCCTTAGCATATGGTGAGCGTGTGCCAGGTGAGCAGGAGCAGCGGGGGATGACTTCCGATCCGACACCCCCAAAATCTCCTGGACAAAGCTGCTGTCTTTGCGGGATAAACAGAAAAGGGCCAGTCCCCACCTCACCCCCAGCCCGCCGCCCCGCAAGTACCTGGGGCTGGGGAGTCAGTGAACTCTCTTCAGCTGTTCGGCTCTCCCGGCTCAGAGCGAGGGGAATCGAGGAGACTGGGCGCAGGATGGGGGTGGACACCCGGCCGCTGCTCCTCCGCGCGGGTAAGTGTGAGCCCCGGGGTGCGGGGAACCGAGCCAGGGACCAGTGACCGCGAGCCGCCGATCCTCCCGCGCTCCCGCGCGCGCGGCCTGCCTTCCCACTGGCTGGCAGAGCACGTCCTCTCGCGCCCGGGGCCTTTGTAAAGAGCGCAGCGGTGGCGCGGAGGTTTTCAGGCTCCGCCCGCTCCACTCGCGCTCCCGCCCCTTCCTTCCTCTCGAGGTGACCCGGCAGCTGGCGCCTTTCTCACCAGGTACCCAGCCCCCTGGCCGCCTCTCTCAGTTCTCTGGCAGGGTTTGGTGCGCG")
 
tokens = tokenizer.tokenize(raw_seq)
lime_string = " ".join(tokens)


start_time = time.time()
explainer = LimeTextExplainer(
    split_expression=r"\s+",
    bow=False,
    class_names=list(model.config.id2label.values())
)

exp = explainer.explain_instance(
    lime_string,
    predict_proba,
    num_features=15,
    num_samples=4000
)



lime_results = exp.as_list()
logger.info(f"LIME explanation results: {lime_results}")

# Convert to DataFrame
df = pd.DataFrame(lime_results, columns=["Token", "Importance"])

# Sort: positive first, then negative
df_sorted = pd.concat([
    df[df["Importance"] > 0].sort_values("Importance", ascending=False),
    df[df["Importance"] < 0].sort_values("Importance")
])

# Save DataFrame
csv_path = "lime_importance_sorted.csv"
df_sorted.to_csv(csv_path, index=False)
logger.info(f"✅ LIME sorted importance saved to: {csv_path}")

# Plot and save figure
plt.figure(figsize=(8, 6))
colors = df_sorted["Importance"].apply(lambda x: "green" if x > 0 else "red")
plt.barh(df_sorted["Token"], df_sorted["Importance"], color=colors)
plt.title("LIME Explanation: Green First")
plt.xlabel("Importance")
plt.gca().invert_yaxis()
plt.tight_layout()

img_path = "lime_explanation_custom_sorted.png"
plt.savefig(img_path, dpi=300)
plt.close()
logger.info(f"✅ Custom sorted LIME plot saved as: {img_path}")
end_time = time.time()

logger.info(f"🕒 Total runtime: {end_time - start_time:.2f} seconds")