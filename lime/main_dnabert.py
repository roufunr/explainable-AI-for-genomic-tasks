import time
import os
import logging
import argparse

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from lime.lime_text import LimeTextExplainer

# ------------------- ARG PARSING -------------------
parser = argparse.ArgumentParser(description="LIME explainability for DNABERT")
parser.add_argument("--job_id",    type=int,   required=True)
parser.add_argument("--expId",     type=str,   required=True)
parser.add_argument("--start_idx", type=int,   required=True)
args = parser.parse_args()

# ------------------- PATH SETUP -------------------
home_dir   = os.path.expanduser("~") + "/data"
log_dir    = f"{home_dir}/{args.expId}/logs/DNABERT"
result_dir = f"{home_dir}/{args.expId}/results/DNABERT/job_{args.job_id}"
os.makedirs(log_dir,    exist_ok=True)
os.makedirs(result_dir, exist_ok=True)

# ------------------- LOGGING -------------------
logging.basicConfig(
    level    = logging.INFO,
    format   = "%(asctime)s %(levelname)s %(message)s",
    handlers = [
        logging.FileHandler(f"{log_dir}/job_{args.job_id}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ------------------- DEVICE -------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------- LOAD MODEL & TOKENIZER -------------------
ckpt      = f"{home_dir}/promo_detection/DNABERT"
tokenizer = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
model     = AutoModelForSequenceClassification.from_pretrained(
                ckpt, trust_remote_code=True
            ).to(DEVICE).eval()

# ------------------- LOAD DATASET -------------------
df = pd.read_csv(f"{home_dir}/promo_detection/dataset/dataset.csv")

# ------------------- PREDICT PROBA -------------------
def predict_proba(texts, batch_size=128):
    all_probs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Batches", leave=False):
        batch = texts[i : i + batch_size]
        enc   = tokenizer(batch,
                          padding=True,
                          truncation=True,
                          add_special_tokens=True,
                          return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            logits = model(**enc).logits
            probs  = torch.softmax(logits, dim=-1).cpu()
        all_probs.append(probs)
    return torch.cat(all_probs, dim=0).numpy()

# ------------------- RUN LIME FOR ONE SEQ -------------------
def run_exp(raw_seq: str, idx: int):
    logger.info(f"Processing idx={idx}")

    # 1) Model prediction & confidence
    enc = tokenizer(raw_seq, return_tensors="pt", truncation=True, padding=True).to(DEVICE)
    with torch.no_grad():
        logits     = model(**enc).logits
        pred_label = int(logits.argmax(dim=-1).item())
        confidence = logits.softmax(dim=-1).squeeze().tolist()
    logger.info(f"Model prediction={pred_label}, confidence={confidence}")

    # 2) Tokenize into k‑mers & make LIME string
    tokens      = tokenizer.tokenize(raw_seq)
    lime_string = " ".join(tokens)

    # 3) Initialize LIME
    explainer = LimeTextExplainer(
        split_expression=r"\s+",
        bow=False,
        class_names=tuple(model.config.id2label.values())
    )

    # 4) Explain the predicted label
    exp = explainer.explain_instance(
        lime_string,
        predict_proba,
        labels=(pred_label,),
        num_features=100,
        num_samples=4000
    )

    # 5) Build index → weight map
    weight_map = dict(exp.as_map().get(pred_label, []))

    # 6) Pair each original token with its weight (or 0.0)
    feats = [(tokens[i], weight_map.get(i, 0.0)) for i in range(len(tokens))]

    # 7) Sort by absolute weight desc, take top 15
    top15 = sorted(feats, key=lambda x: abs(x[1]), reverse=True)[:15]
    logger.info(f"Top 15 features idx={idx}: {top15}")
    return top15

# ------------------- SAVE CSV & PLOT -------------------
def generate_csv_and_image(feats, idx):
    df_feats = pd.DataFrame(feats, columns=["Token", "Importance"])
    pos = df_feats[df_feats.Importance > 0].sort_values("Importance", ascending=False)
    neg = df_feats[df_feats.Importance < 0].sort_values("Importance")
    df_sorted = pd.concat([pos, neg])

    # CSV
    csv_path = f"{result_dir}/{idx}.csv"
    df_sorted.to_csv(csv_path, index=False)
    logger.info(f"Saved CSV → {csv_path}")

    # Plot
    plt.figure(figsize=(8,6))
    colors = ["green" if v > 0 else "red" for v in df_sorted.Importance]
    plt.barh(df_sorted.Token, df_sorted.Importance, color=colors)
    plt.title("Top 15 LIME Feature Importances")
    plt.xlabel("Importance")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    img_path = f"{result_dir}/{idx}.png"
    plt.savefig(img_path, dpi=300)
    plt.close()
    logger.info(f"Saved plot → {img_path}")

# ------------------- MAIN LOOP -------------------
def main():
    for idx in range(args.start_idx, args.start_idx + 10):
        seq   = df.loc[idx, "sequence"]
        feats = run_exp(seq, idx)
        generate_csv_and_image(feats, idx)

if __name__ == "__main__":
    main()
