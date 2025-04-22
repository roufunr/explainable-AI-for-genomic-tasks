import time
import os
import logging
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import shap
from tqdm.auto import tqdm



parser = argparse.ArgumentParser()
parser.add_argument("--job_id", type=int, default=1)
parser.add_argument("--expId", type=str, default='SHAP_exp')
parser.add_argument("--start_idx", type=int, default=10)
parser.add_argument("--n_seq", type=int, default=50)
args = parser.parse_args()

home_dir = "/lustre/fs1/home/sbiswas/ML/project/SHAP/"

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

class ShapExplainer:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        
    def _predict_proba_batch(self, texts, batch_size=32):
        """Make predictions for a batch of texts."""
        all_probs = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            
            inputs = self.tokenizer(batch, 
                                  padding=True, 
                                  truncation=True, 
                                  return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs).logits
                probs = torch.softmax(outputs, dim=-1).cpu().numpy()
                
            all_probs.append(probs)
            
        return np.vstack(all_probs)
    
    def explain_instance(self, text, num_features=15, num_samples=1000):
        """Explain a single text instance using SHAP."""
        logger.info("Tokenizing input text")
        tokens = self.tokenizer.tokenize(text)
        logger.info(f"Tokenized into {len(tokens)} tokens")
        
        # For genomic data, we'll use perturbation-based approach
        # First, define the prediction function that takes tokenized text
        def f(token_presence):
            """
            Prediction function for SHAP that takes binary indicators
            of token presence (1 = present, 0 = absent)
            """
            # Convert binary indicators to actual texts by masking tokens
            masked_texts = []
            for mask in token_presence:
                # Create masked version of tokens
                current_tokens = [t if m else "" for t, m in zip(tokens, mask)]
                # Join tokens and add to batch
                masked_texts.append(" ".join([t for t in current_tokens if t]))
                
            # Run prediction on the masked texts
            return self._predict_proba_batch(masked_texts)
        
        # Create binary data for the background (all possible token combinations would be 2^n,
        # so we sample a reasonable number)
        np.random.seed(42)
        num_bg_samples = min(100, 2**len(tokens))  # Cap at 100 samples
        
        logger.info(f"Generating {num_bg_samples} background samples")
        # Create background dataset as numpy array - each row represents token presence
        background = np.zeros((num_bg_samples, len(tokens)), dtype=np.bool_)
        for i in range(num_bg_samples):
            # Random binary mask (0 = absent, 1 = present)
            background[i] = np.random.choice([0, 1], size=len(tokens), p=[0.3, 0.7])
        
        # For our main instance, all tokens are present
        instance = np.ones(len(tokens), dtype=np.bool_)
        
        logger.info("Creating SHAP explainer")
        # Create the KernelExplainer with our function and background
        explainer = shap.KernelExplainer(f, background)
        
        logger.info(f"Calculating SHAP values with {num_samples} samples")
        # Get SHAP values for the instance
        shap_values = explainer.shap_values(
            instance.reshape(1, -1),
            nsamples=num_samples
        )
        
        # # Determine the predicted class for the full instance
        # pred_probs = f(instance.reshape(1, -1))
        # pred_class = np.argmax(pred_probs[0])
        pred_class = 1
        logger.info(f"Predicted class: {pred_class}")
        
        # If shap_values is a list (one array per class), select the array for predicted class
        if isinstance(shap_values, list):
            logger.info(f"SHAP returned values for {len(shap_values)} classes")
            shap_values_for_class = shap_values[pred_class][0]
        else:
            logger.info("SHAP returned a single array of values")
            shap_values_for_class = shap_values[0]
        
        # Map SHAP values to tokens
        token_importances = []
        for i, token in enumerate(tokens):
            if i < len(shap_values_for_class):
                # print("----------->", shap_values_for_class[i])
                importance = shap_values_for_class[i][0].item()    # float(shap_values_for_class[i])
                token_importances.append((token, importance))
                logger.debug(f"Token: {token}, Importance: {importance}")
                
        
        # Sort by absolute importance and take top features
        token_importances.sort(key=lambda x: abs(x[1]), reverse=True)
        
        return token_importances[:num_features]


def run_exp(raw_seq, idx):
    logger.info(f"Running experiment for {idx}::sequence:{raw_seq[:50]}...")
    
    # Create explainer
    explainer = ShapExplainer(model, tokenizer)
    
    # Get SHAP values
    start_time = time.time()
    shap_results = explainer.explain_instance(raw_seq, num_features=15, num_samples=1000)
    logger.info(f"SHAP calculation took {time.time() - start_time:.2f} seconds")
    
    return shap_results

def generate_csv_and_image(shap_results, idx): 
    logger.info(f"SHAP explanation results: {shap_results}")

    # Convert to DataFrame
    df = pd.DataFrame(shap_results, columns=["Token", "Importance"])

    # Sort: positive first, then negative
    df_sorted = pd.concat([
        df[df["Importance"] > 0].sort_values("Importance", ascending=False),
        df[df["Importance"] < 0].sort_values("Importance")
    ])

    # Save DataFrame
    csv_path = f"{RESULT_DIR}/{idx}.csv"
    df_sorted.to_csv(csv_path, index=False)
    logger.info(f"✅ SHAP sorted importance saved to: {csv_path}")

    # Plot and save figure
    plt.figure(figsize=(8, 6))
    colors = df_sorted["Importance"].apply(lambda x: "green" if x > 0 else "red")
    plt.barh(df_sorted["Token"], df_sorted["Importance"], color=colors)
    plt.title("SHAP Explanation: Green First")
    plt.xlabel("Importance")
    plt.gca().invert_yaxis()
    plt.tight_layout()

    img_path = f"{RESULT_DIR}/{idx}.png"
    plt.savefig(img_path, dpi=300)
    plt.close()
    logger.info(f"✅ Custom sorted SHAP plot saved as: {img_path}")
    
    # Generate waterfall plot
    try:
        plt.figure(figsize=(12, 8))
        
        # Sort by importance for waterfall plot
        waterfall_data = df_sorted.copy()
        
        # Create waterfall chart
        base = 0
        y_pos = np.arange(len(waterfall_data))
        
        # Plot bars
        for i, (_, row) in enumerate(waterfall_data.iterrows()):
            plt.barh(i, row['Importance'], left=base if row['Importance'] < 0 else base, 
                    color='green' if row['Importance'] > 0 else 'red')
            
        # Add token labels
        plt.yticks(y_pos, waterfall_data['Token'])
        plt.title("SHAP Waterfall Plot")
        plt.xlabel("SHAP Value (Impact on Model Output)")
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        waterfall_path = f"{RESULT_DIR}/{idx}_waterfall.png"
        plt.savefig(waterfall_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"✅ SHAP waterfall plot saved as: {waterfall_path}")
    except Exception as e:
        logger.error(f"Failed to generate waterfall plot: {e}")

def main():
    startIdx = args.start_idx
    n_seq = args.n_seq

    for idx in range(startIdx, startIdx + n_seq):
        start_time = time.time()
        sequence = dataset.iloc[idx]["sequence"]
        shap_results = run_exp(sequence, idx)
        generate_csv_and_image(shap_results, idx)
        end_time = time.time()
        logger.info(f"{idx} took {end_time - start_time} seconds")
        # break
    
if __name__ == "__main__":
    main()