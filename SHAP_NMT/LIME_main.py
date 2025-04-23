import time
import os
import logging
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch, numpy as np
from lime.lime_text import LimeTextExplainer
from tqdm.auto import tqdm   
import matplotlib.pyplot as plt
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--job_id", type=int, default=0)            # required=True)
parser.add_argument("--expId", type=str, default='LIME_exp')      # required=True)
parser.add_argument("--start_idx", type=int, default=0)         # required=True)
args = parser.parse_args()

home_dir = "/home/sagor/sagor/CAP_5610/project/SHAP/"       #f"{os.path.expanduser("~")}/data"
print("Home Directory is = ", home_dir)

# ---------------------------LOGGER-------------------------------------
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
# ----------------------------------------------------------------

#-------------------------------MODEL AND DATASET----------------------------
ckpt = f"{home_dir}/NMT/"
dataset_path = f"{home_dir}/dataset.csv"
dataset = pd.read_csv(dataset_path)
#----------------------------------------------------------------------------


device = torch.device("cuda")
tokenizer = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
model = (AutoModelForSequenceClassification
         .from_pretrained(ckpt, trust_remote_code=True)
         .to(device)
         .eval())

def predict_proba(bpe_strings, batch_size=128):
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

def run_exp(raw_seq, idx):
    # raw_seq = ("CTGAAAGCTGAATACAGAAGGCATAGATGCTGCATCTTGAGTGTCCAGCTCTTCTGTGCTGGACATCAGGATGTATACACTTAACCTGGGAGGCTTAGAGTTGGGAAGAGAAGAGTTAGATCGAGGGCAAGGCTTCTGACTCCTCTCTGGGGAAAGAAGAATGGATACGTGTAGAGGTGCTATGTGCAAAAACAGCACCTCCACCTCTGCTACCCTCCCTGCAACACAACACACACACACACACACACACACGTACACACACACTCACTAAGGTCCTGCAAAGCCCGTGAAGAGACATACCACCTCTCCTTGCCAAAGGGTGTTCGGCCTTAAAATGCCATAAACAAAAACATATACAGAAATATTTTCGCTACAAGGACCCAGATGTCTAAGCAGTTTAACAGAAAGCTTGGCACCCCTGGGTGGTCATGCATGCCAACCAGATTTCTCATGCTTTAAGAAGTCTATTTTTTAAAACAGGATTGACCAAACTAATGGGGTCTGCTCTGGATGGTGAAAATGTTTACTTTCATTTCCCCCCAACCCCCTCGCCTCCCCAGTTCCCCCTGAATATATTCTCAAGAAAGAGTATGTCTTTGGGTGGTTGCAAGTAGGAAGTTTCAAGGCTTTTCTCCCTAACGAATACCGAACACTGAGCACACAGGAGGGGAACAGAAACTTGGAAGCAGAGCAAAGGACCAAGTTATCCTCCTGTCTAACCCTGCTCAACAGGATCACCCTGAAGTGTGGCTCTGAAGATATTTGTGAACTCTGGCACAGCTGTTTGGCAGAATTAATACTAAAACCACAGGTCTTGCAAAAACAAAGAATCAAAAAAAAAAAAAAAAAGTGAAAACCAAACAGCAGGAGGAAACCCTTAGGAAAGCTCGGTGTCTTTCAGACATGAGATTTTCTTTCTCCCACTCTACTTGACTGGCCTGACCTGGGGTCCAGTCCCACTTCCTGGTGGCTGCGGAAATCCGCAGTGAGGCTCAGTGTGGATTTTGGCTTCAGAGGATTTGGGAAATTCCACCTTTCAATCTGAAATCTGGGGAAAGTCCCAGAAAACAACGGATTACTATATTTCTCTCATGAACCAGACTGAAGCAAGGAGTCAGGTCTAGGACCAAGGACCTCGGCACTCCAGGGAGCCCAGGGCCCCTCAGAAAGGTCTCGCTTGGCCACCGAGAGCGTGGGGCCGTAGCAGTTGGCAAGCGGGGGAGGGGGGTCACGCACAACTGCTCTGGAGCCACACTGCATAAATGGCATTCGTTTTTGTTTTCTGATTAGAAAAACGTTGAAAAGAATACCAAAGTACAGAGAAAACACACCCCCATCTCCAATGTTTCCTCATTTATGTAAATATTCTCATATTTTATAAAACAGGGAGGTGCTGTCTATAATTTTTGCAGGCTTTCCTTAGCATATGGTGAGCGTGTGCCAGGTGAGCAGGAGCAGCGGGGGATGACTTCCGATCCGACACCCCCAAAATCTCCTGGACAAAGCTGCTGTCTTTGCGGGATAAACAGAAAAGGGCCAGTCCCCACCTCACCCCCAGCCCGCCGCCCCGCAAGTACCTGGGGCTGGGGAGTCAGTGAACTCTCTTCAGCTGTTCGGCTCTCCCGGCTCAGAGCGAGGGGAATCGAGGAGACTGGGCGCAGGATGGGGGTGGACACCCGGCCGCTGCTCCTCCGCGCGGGTAAGTGTGAGCCCCGGGGTGCGGGGAACCGAGCCAGGGACCAGTGACCGCGAGCCGCCGATCCTCCCGCGCTCCCGCGCGCGCGGCCTGCCTTCCCACTGGCTGGCAGAGCACGTCCTCTCGCGCCCGGGGCCTTTGTAAAGAGCGCAGCGGTGGCGCGGAGGTTTTCAGGCTCCGCCCGCTCCACTCGCGCTCCCGCCCCTTCCTTCCTCTCGAGGTGACCCGGCAGCTGGCGCCTTTCTCACCAGGTACCCAGCCCCCTGGCCGCCTCTCTCAGTTCTCTGGCAGGGTTTGGTGCGCG")
    logger.info(f"running experiment for {idx}::sequence:{raw_seq}")
    tokens = tokenizer.tokenize(raw_seq)
    lime_string = " ".join(tokens)

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
    return exp.as_list()

def generate_csv_and_image(lime_results, idx): 
    logger.info(f"LIME explanation results: {lime_results}")

    # Convert to DataFrame
    df = pd.DataFrame(lime_results, columns=["Token", "Importance"])

    # Sort: positive first, then negative
    df_sorted = pd.concat([
        df[df["Importance"] > 0].sort_values("Importance", ascending=False),
        df[df["Importance"] < 0].sort_values("Importance")
    ])

    # Save DataFrame
    csv_path = f"{RESULT_DIR}/{idx}.csv"
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

    img_path = f"{RESULT_DIR}/{idx}.png"
    plt.savefig(img_path, dpi=300)
    plt.close()
    logger.info(f"✅ Custom sorted LIME plot saved as: {img_path}")
    
def main():
    startIdx = args.start_idx
    sequence = ("CTGAAAGCTGAATACAGAAGGCATAGATGCTGCATCTTGAGTGTCCAGCTCTTCTGTGCTGGACATCAGGATGTATACACTTAACCTGGGAGGCTTAGAGTTGGGAAGAGAAGAGTTAGATCGAGGGCAAGGCTTCTGACTCCTCTCTGGGGAAAGAAGAATGGATACGTGTAGAGGTGCTATGTGCAAAAACAGCACCTCCACCTCTGCTACCCTCCCTGCAACACAACACACACACACACACACACACACGTACACACACACTCACTAAGGTCCTGCAAAGCCCGTGAAGAGACATACCACCTCTCCTTGCCAAAGGGTGTTCGGCCTTAAAATGCCATAAACAAAAACATATACAGAAATATTTTCGCTACAAGGACCCAGATGTCTAAGCAGTTTAACAGAAAGCTTGGCACCCCTGGGTGGTCATGCATGCCAACCAGATTTCTCATGCTTTAAGAAGTCTATTTTTTAAAACAGGATTGACCAAACTAATGGGGTCTGCTCTGGATGGTGAAAATGTTTACTTTCATTTCCCCCCAACCCCCTCGCCTCCCCAGTTCCCCCTGAATATATTCTCAAGAAAGAGTATGTCTTTGGGTGGTTGCAAGTAGGAAGTTTCAAGGCTTTTCTCCCTAACGAATACCGAACACTGAGCACACAGGAGGGGAACAGAAACTTGGAAGCAGAGCAAAGGACCAAGTTATCCTCCTGTCTAACCCTGCTCAACAGGATCACCCTGAAGTGTGGCTCTGAAGATATTTGTGAACTCTGGCACAGCTGTTTGGCAGAATTAATACTAAAACCACAGGTCTTGCAAAAACAAAGAATCAAAAAAAAAAAAAAAAAGTGAAAACCAAACAGCAGGAGGAAACCCTTAGGAAAGCTCGGTGTCTTTCAGACATGAGATTTTCTTTCTCCCACTCTACTTGACTGGCCTGACCTGGGGTCCAGTCCCACTTCCTGGTGGCTGCGGAAATCCGCAGTGAGGCTCAGTGTGGATTTTGGCTTCAGAGGATTTGGGAAATTCCACCTTTCAATCTGAAATCTGGGGAAAGTCCCAGAAAACAACGGATTACTATATTTCTCTCATGAACCAGACTGAAGCAAGGAGTCAGGTCTAGGACCAAGGACCTCGGCACTCCAGGGAGCCCAGGGCCCCTCAGAAAGGTCTCGCTTGGCCACCGAGAGCGTGGGGCCGTAGCAGTTGGCAAGCGGGGGAGGGGGGTCACGCACAACTGCTCTGGAGCCACACTGCATAAATGGCATTCGTTTTTGTTTTCTGATTAGAAAAACGTTGAAAAGAATACCAAAGTACAGAGAAAACACACCCCCATCTCCAATGTTTCCTCATTTATGTAAATATTCTCATATTTTATAAAACAGGGAGGTGCTGTCTATAATTTTTGCAGGCTTTCCTTAGCATATGGTGAGCGTGTGCCAGGTGAGCAGGAGCAGCGGGGGATGACTTCCGATCCGACACCCCCAAAATCTCCTGGACAAAGCTGCTGTCTTTGCGGGATAAACAGAAAAGGGCCAGTCCCCACCTCACCCCCAGCCCGCCGCCCCGCAAGTACCTGGGGCTGGGGAGTCAGTGAACTCTCTTCAGCTGTTCGGCTCTCCCGGCTCAGAGCGAGGGGAATCGAGGAGACTGGGCGCAGGATGGGGGTGGACACCCGGCCGCTGCTCCTCCGCGCGGGTAAGTGTGAGCCCCGGGGTGCGGGGAACCGAGCCAGGGACCAGTGACCGCGAGCCGCCGATCCTCCCGCGCTCCCGCGCGCGCGGCCTGCCTTCCCACTGGCTGGCAGAGCACGTCCTCTCGCGCCCGGGGCCTTTGTAAAGAGCGCAGCGGTGGCGCGGAGGTTTTCAGGCTCCGCCCGCTCCACTCGCGCTCCCGCCCCTTCCTTCCTCTCGAGGTGACCCGGCAGCTGGCGCCTTTCTCACCAGGTACCCAGCCCCCTGGCCGCCTCTCTCAGTTCTCTGGCAGGGTTTGGTGCGCG")
    
    for idx in range(startIdx, startIdx + 10):
        start_time = time.time()
        sequence = dataset.iloc[idx]["sequence"]
        lime_results = run_exp(sequence, idx)
        generate_csv_and_image(lime_results, idx)
        end_time = time.time()
        logger.info(f"{idx} took {end_time - start_time} seconds")
        break
    
if __name__ == "__main__":
    main()