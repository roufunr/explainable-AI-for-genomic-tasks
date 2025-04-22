import time
import os
import logging
import sys
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch, numpy as np
import pandas as pd
from tqdm.auto import tqdm
import random

print("Debug: Starting script")
ckpt = f"NMT"
dataset_path = f"dataset.csv"
dataset = pd.read_csv(dataset_path)

device = torch.device("cuda")
tokenizer = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
model = (AutoModelForSequenceClassification
         .from_pretrained(ckpt, trust_remote_code=True)
         .to(device)
         .eval())


def predict_proba(bpe_strings, batch_size=128):
    """Anchor → probs on GPU, with a tqdm bar."""
    probs = []
    print(len(bpe_strings))
    for i in tqdm(range(0, len(bpe_strings), batch_size), desc="Anchor batches", leave=False):
        batch = bpe_strings[i:i+batch_size]

        enc = tokenizer(batch,
                        padding=True,
                        truncation=True,
                        add_special_tokens=True,
                        return_tensors="pt").to(device)

        with torch.no_grad():
            logits = model(**enc).logits
            probs.append(torch.softmax(logits, dim=-1).cpu())
    print(len(probs))
    return torch.cat(probs).numpy()


# Simplified Anchor implementation
class SimpleAnchor:
    def __init__(self, class_names):
        self.class_names = class_names
    
    def explain_instance(self, text, predict_fn, num_features=10, num_samples=20, max_tokens=100):
        """
        A simplified implementation of Anchor for text data.
        
        Args:
            text: The text to explain
            predict_fn: A function that takes a list of texts and returns class probabilities
            num_features: Number of features to include in the explanation
            num_samples: Number of samples to use for estimating precision
            max_tokens: Maximum number of tokens to analyze (for efficiency)
            
        Returns:
            A dictionary with the explanation
        """
        # Split the text into tokens
        tokens = text.split()
        
        # If there are too many tokens, take a subset
        if len(tokens) > max_tokens:
            # print(f"Debug: Limiting analysis to {max_tokens} tokens out of {len(tokens)}")
            # Take tokens at regular intervals to cover the whole sequence
            step = len(tokens) // max_tokens
            subset_indices = [i * step for i in range(max_tokens)]
            subset_tokens = [tokens[i] for i in subset_indices]
            # Create a new text with only the subset of tokens
            subset_text = " ".join(subset_tokens)
            tokens = subset_tokens
            text = subset_text
        
        # print(f"Debug: Analyzing {len(tokens)} tokens")
        
        # Get the predicted class for the original text
        original_probs = predict_fn([text])[0]
        original_pred = np.argmax(original_probs)
        # print(f"Debug: Original prediction: class {original_pred} with probability {original_probs[original_pred]:.4f}")
        
        # Calculate feature importance by removing each token and measuring the change in prediction
        # print("Debug: Calculating feature importance...")
        importances = []
        for i in range(len(tokens)):
           
            # Create a new text with the token removed
            new_tokens = tokens.copy()
            removed_token = new_tokens.pop(i)
            new_text = " ".join(new_tokens)
            
            # Get the prediction for the new text
            new_pred = predict_fn([new_text])[0]
            
            # Calculate the importance as the change in probability for the original class
            importance = abs(new_pred[original_pred] - original_probs[original_pred])
            importances.append((i, removed_token, importance))
        
        # Sort the tokens by importance
        importances.sort(key=lambda x: x[2], reverse=True)
        
        # Select the top tokens as the explanation
        top_tokens = [imp[1] for imp in importances[:num_features]]
        
        # Store the importance scores of the top tokens
        top_importances = [imp[2] for imp in importances[:num_features]]
        
        # Calculate precision by sampling
        precision = self._calculate_precision(text, top_tokens, predict_fn, original_pred, num_samples)
        
        # Calculate coverage
        coverage = len(top_tokens) / len(tokens)
        
        return {
            'tokens': top_tokens,
            'importances': top_importances,
            'precision': precision,
            'coverage': coverage,
            'class': self.class_names[original_pred]
        }
    
    def _calculate_precision(self, text, top_tokens, predict_fn, original_pred, num_samples):
        """
        Calculate the precision of the explanation by sampling.
        
        Precision is the fraction of instances where the model's prediction
        is the same as the original prediction when only the top tokens are present.
        """
        tokens = text.split()
        matches = 0
        
        # print(f"Debug: Calculating precision with {num_samples} samples")
        # print(f"Debug: Top tokens: {top_tokens}")
        
        for i in range(num_samples):
            # if i % 5 == 0:
            #     print(f"Debug: Processed {i}/{num_samples} samples")
                
            # Create a new text with only the top tokens and random other tokens
            new_tokens = []
            for token in tokens:
                if token in top_tokens or random.random() < 0.5:
                    new_tokens.append(token)
            
            if not new_tokens:  # Ensure we have at least one token
                new_tokens = top_tokens[:1]
            
            new_text = " ".join(new_tokens)
            
            # Get the prediction for the new text
            new_pred = np.argmax(predict_fn([new_text])[0])
            
            # Check if the prediction matches the original
            if new_pred == original_pred:
                matches += 1
        
        precision = matches / num_samples
        print(f"Debug: Precision calculation complete: {precision:.4f} ({matches}/{num_samples})")
        return precision


# Create the explainer
explainer = SimpleAnchor(class_names=list(model.config.id2label.values()))
# print("Debug: Created SimpleAnchor explainer")

# Define the number of sequences to process (use a small number for testing)
num_sequences = len(dataset)  # Process all sequences in the dataset
# print(f"Debug: Processing {num_sequences} sequences from the dataset")

# Create the CSV file with headers
csv_file = 'result2.csv'
with open(csv_file, 'w') as f:
    f.write('sequence_idx,true_label,predicted_label,anchor,importance_values,precision,coverage,model_accuracy\n')

# Loop through the dataset
for idx in range(num_sequences):
    # print(f"\nDebug: Processing sequence {idx+1}/{num_sequences}")
    
    # Get the sequence and true label
    row = dataset.iloc[idx]
    raw_seq = row['sequence']
    true_label = row['label']
    # print(f"Debug: Sequence index {idx} with true label {true_label}")
    
    # Tokenize the sequence
    tokens = tokenizer.tokenize(raw_seq)
    anchor_string = " ".join(tokens)
    # print(f"Debug: Tokenized sequence (length: {len(tokens)} tokens)")
    
    # Generate explanation
    # print("Debug: Generating explanation")
    explanation = explainer.explain_instance(
        anchor_string,
        predict_proba,
        num_features=10,  # Reduced from 15 to 10
        num_samples=20,   # Reduced from 100 to 20
        max_tokens=50     # Limit to 50 tokens for faster processing
    )
    
    # Extract the numeric part from the class label (e.g., 'LABEL_0' -> 0)
    predicted_label = int(explanation['class'].split('_')[-1])
    model_accuracy = 'Correct' if int(true_label) == predicted_label else 'Incorrect'
    
    # Format the importance values as a string (with 4 decimal places)
    importance_values = ";".join([f"{imp:.4f}" for imp in explanation['importances']])
    
   
    # Create a result dictionary
    result = {
        'sequence_idx': idx,
        'true_label': true_label,
        'predicted_label': predicted_label,
        'anchor': ' AND '.join(explanation['tokens']),
        'importance_values': importance_values,
        'precision': explanation['precision'],
        'coverage': explanation['coverage'],
        'model_accuracy': model_accuracy
    }
    
    # Write the result to the CSV file
    with open(csv_file, 'a') as f:
        f.write(f"{result['sequence_idx']},{result['true_label']},{result['predicted_label']},\"{result['anchor']}\",\"{result['importance_values']}\",{result['precision']},{result['coverage']},{result['model_accuracy']}\n")
    
    print(f"Debug: Result for sequence {idx} saved to {csv_file}")

print(f"\nDebug: All results saved to {csv_file}")
print("Debug: Script completed")
