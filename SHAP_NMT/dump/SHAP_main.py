
import time
import os
import logging
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
import shap
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import argparse

from shap import maskers, Explanation

parser = argparse.ArgumentParser()
parser.add_argument("--job_id", type=int, default=0)            # required=True)
parser.add_argument("--expId", type=str, default='SHAP_exp')      # required=True)
parser.add_argument("--start_idx", type=int, default=0)         # required=True)
args = parser.parse_args()


home_dir = "/home/sagor/sagor/CAP_5610/project/SHAP/"       #f"{os.path.expanduser("~")}/data"
# print("Home Directory is = ", home_dir)

LOG_DIR = f"{home_dir}/{args.expId}/logs"
LOG_FILE = os.path.join(LOG_DIR, f"job_{args.job_id}.log")
os.makedirs(LOG_DIR, exist_ok=True)

RESULT_DIR = f"{home_dir}/{args.expId}/results/job_{args.job_id}"
os.makedirs(RESULT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ------------------ Load model and tokenizer ------------------
ckpt = f"{home_dir}/NMT/"
dataset_path = f"{home_dir}/dataset.csv"
dataset = pd.read_csv(dataset_path)

device = torch.device("cuda")
tokenizer = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
model = AutoModelForSequenceClassification.from_pretrained(ckpt, trust_remote_code=True).to(device).eval()

# # ------------------ SHAP Wrapper Function ---------------------
# def shap_predict(input_ids_numpy):
#     """Convert numpy input_ids to model prediction probabilities."""
#     input_ids_tensor = torch.tensor(input_ids_numpy).long().to(device)
#     attention_mask = (input_ids_tensor != tokenizer.pad_token_id).long()
#     with torch.no_grad():
#         outputs = model(input_ids=input_ids_tensor, attention_mask=attention_mask)
#         probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
#     return probs[:, 1].cpu().numpy()  # Probability of promoter class (label=1)


def shap_predict(texts):
    """Accept list of raw DNA strings"""
    enc = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**enc).logits
        probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs[:, 1].cpu().numpy()  # Probability of class 1




# # ------------------ SHAP Experiment Function -------------------
# def run_shap_exp(raw_seq, idx):
#     logger.info(f"Running SHAP for idx {idx}:: sequence: {raw_seq[:25]}...")

#     # Tokenize and encode
#     encoding = tokenizer(raw_seq, return_tensors="pt", truncation=True)
#     input_ids = encoding["input_ids"].to(device)

#     # Prepare background data (1 sequence is okay for transformer)
#     background = input_ids[:1].cpu().numpy()

#     # Create SHAP explainer
#     explainer = shap.Explainer(shap_predict, background)

#     # Explain current sequence
#     shap_values = explainer(input_ids.cpu().numpy())

#     print("Original prediction:", shap_predict(input_ids.cpu().numpy()))

#     # Zero out half the tokens (simulate masking)
#     masked_ids = input_ids.clone()
#     masked_ids[0, 10:30] = tokenizer.pad_token_id
#     print("Masked prediction:", shap_predict(masked_ids.cpu().numpy()))


#     # Convert to readable tokens
#     tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

#     # Create Explanation object manually
#     explanation = shap.Explanation(
#         values=shap_values[0].values,
#         data=tokens,
#         base_values=shap_values[0].base_values
#     )

#     # Save SHAP values
#     token_scores = list(zip(tokens, shap_values[0].values))
#     shap_csv_path = os.path.join(RESULT_DIR, f"shap_scores_{idx}.csv")
#     pd.DataFrame(token_scores, columns=["token", "shap_value"]).to_csv(shap_csv_path, index=False)

#     # Optional: Plot and save SHAP visualization
#     plt.figure()
#     # shap.plots.text(shap_values[0], display=False)
#     # shap.plots.text(shap.Explanation(values=shap_values[0].values, data=tokens))
#     shap.plots.text(explanation, display=False)
#     plt.savefig(os.path.join(RESULT_DIR, f"shap_plot_{idx}.png"))
#     plt.close()

#     return token_scores


def run_shap_exp(raw_seq, idx):
    logger.info(f"Running SHAP for idx {idx}:: sequence: {raw_seq[:60]}...")

    # Use SHAP's tokenizer-aware Text masker
    text_masker = maskers.Text(tokenizer)

    # Use explainer with raw text input
    explainer = shap.Explainer(shap_predict, text_masker)

    # Run explanation on original raw sequence
    shap_values = explainer([raw_seq])  # Note: now passing list of string

    # Tokens and values
    tokens = shap_values[0].data
    values = shap_values[0].values

    # Save CSV
    token_scores = list(zip(tokens, values))
    shap_csv_path = os.path.join(RESULT_DIR, f"shap_scores_{idx}.csv")
    pd.DataFrame(token_scores, columns=["token", "shap_value"]).to_csv(shap_csv_path, index=False)

    # Visualize
    explanation = Explanation(values=values, data=tokens, base_values=shap_values[0].base_values)
    plt.figure()
    shap.plots.text(explanation, display=False)
    plt.savefig(os.path.join(RESULT_DIR, f"shap_plot_{idx}.png"))
    plt.close()

    return token_scores



# ------------------ Main Batch Driver ------------------------
def main():
    startIdx = args.start_idx
    sequence = ("CTGAAAGCTGAATACAGAAGGCATAGATGCTGCATCTTGAGTGTCCAGCTCTTCTGTGCTGGACATCAGGATGTATACACTTAACCTGGGAGGCTTAGAGTTGGGAAGAGAAGAGTTAGATCGAGGGCAAGGCTTCTGACTCCTCTCTGGGGAAAGAAGAATGGATACGTGTAGAGGTGCTATGTGCAAAAACAGCACCTCCACCTCTGCTACCCTCCCTGCAACACAACACACACACACACACACACACACGTACACACACACTCACTAAGGTCCTGCAAAGCCCGTGAAGAGACATACCACCTCTCCTTGCCAAAGGGTGTTCGGCCTTAAAATGCCATAAACAAAAACATATACAGAAATATTTTCGCTACAAGGACCCAGATGTCTAAGCAGTTTAACAGAAAGCTTGGCACCCCTGGGTGGTCATGCATGCCAACCAGATTTCTCATGCTTTAAGAAGTCTATTTTTTAAAACAGGATTGACCAAACTAATGGGGTCTGCTCTGGATGGTGAAAATGTTTACTTTCATTTCCCCCCAACCCCCTCGCCTCCCCAGTTCCCCCTGAATATATTCTCAAGAAAGAGTATGTCTTTGGGTGGTTGCAAGTAGGAAGTTTCAAGGCTTTTCTCCCTAACGAATACCGAACACTGAGCACACAGGAGGGGAACAGAAACTTGGAAGCAGAGCAAAGGACCAAGTTATCCTCCTGTCTAACCCTGCTCAACAGGATCACCCTGAAGTGTGGCTCTGAAGATATTTGTGAACTCTGGCACAGCTGTTTGGCAGAATTAATACTAAAACCACAGGTCTTGCAAAAACAAAGAATCAAAAAAAAAAAAAAAAAGTGAAAACCAAACAGCAGGAGGAAACCCTTAGGAAAGCTCGGTGTCTTTCAGACATGAGATTTTCTTTCTCCCACTCTACTTGACTGGCCTGACCTGGGGTCCAGTCCCACTTCCTGGTGGCTGCGGAAATCCGCAGTGAGGCTCAGTGTGGATTTTGGCTTCAGAGGATTTGGGAAATTCCACCTTTCAATCTGAAATCTGGGGAAAGTCCCAGAAAACAACGGATTACTATATTTCTCTCATGAACCAGACTGAAGCAAGGAGTCAGGTCTAGGACCAAGGACCTCGGCACTCCAGGGAGCCCAGGGCCCCTCAGAAAGGTCTCGCTTGGCCACCGAGAGCGTGGGGCCGTAGCAGTTGGCAAGCGGGGGAGGGGGGTCACGCACAACTGCTCTGGAGCCACACTGCATAAATGGCATTCGTTTTTGTTTTCTGATTAGAAAAACGTTGAAAAGAATACCAAAGTACAGAGAAAACACACCCCCATCTCCAATGTTTCCTCATTTATGTAAATATTCTCATATTTTATAAAACAGGGAGGTGCTGTCTATAATTTTTGCAGGCTTTCCTTAGCATATGGTGAGCGTGTGCCAGGTGAGCAGGAGCAGCGGGGGATGACTTCCGATCCGACACCCCCAAAATCTCCTGGACAAAGCTGCTGTCTTTGCGGGATAAACAGAAAAGGGCCAGTCCCCACCTCACCCCCAGCCCGCCGCCCCGCAAGTACCTGGGGCTGGGGAGTCAGTGAACTCTCTTCAGCTGTTCGGCTCTCCCGGCTCAGAGCGAGGGGAATCGAGGAGACTGGGCGCAGGATGGGGGTGGACACCCGGCCGCTGCTCCTCCGCGCGGGTAAGTGTGAGCCCCGGGGTGCGGGGAACCGAGCCAGGGACCAGTGACCGCGAGCCGCCGATCCTCCCGCGCTCCCGCGCGCGCGGCCTGCCTTCCCACTGGCTGGCAGAGCACGTCCTCTCGCGCCCGGGGCCTTTGTAAAGAGCGCAGCGGTGGCGCGGAGGTTTTCAGGCTCCGCCCGCTCCACTCGCGCTCCCGCCCCTTCCTTCCTCTCGAGGTGACCCGGCAGCTGGCGCCTTTCTCACCAGGTACCCAGCCCCCTGGCCGCCTCTCTCAGTTCTCTGGCAGGGTTTGGTGCGCG")
    for idx in range(startIdx, startIdx + 10):
        start_time = time.time()
        #sequence = dataset.iloc[idx]["sequence"]
        token_scores = run_shap_exp(sequence, idx)
        end_time = time.time()
        logger.info(f"{idx} took {end_time - start_time:.2f} seconds")
        break

if __name__ == "__main__":
    main()




