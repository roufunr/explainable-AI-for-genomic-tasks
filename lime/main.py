from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch, numpy as np
from lime.lime_text import LimeTextExplainer
from tqdm.auto import tqdm                     # progress bar
 
ckpt   = "/home/sourav/promoter_detection/codes/DNABERT2/output/dnabert2_promoter_detection/checkpoint-4600"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
# tokenizer = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
# model     = (AutoModelForSequenceClassification
#              .from_pretrained(ckpt, trust_remote_code=True)
#              .to(device)
#              .eval())

# def predict_proba(bpe_strings, batch_size=256):
#     """LIME → probs on GPU, with a tqdm bar."""
#     probs = []
#     for i in tqdm(range(0, len(bpe_strings), batch_size),
#                   desc="LIME batches", leave=False):
#         batch = bpe_strings[i:i+batch_size]
 
#         enc = tokenizer(batch,
#                         padding=True,
#                         truncation=True,
#                         add_special_tokens=True,   # CLS/SEP go back in
#                         return_tensors="pt").to(device)
 
#         with torch.no_grad():
#             logits = model(**enc).logits
#             probs.append(torch.softmax(logits, dim=-1).cpu())
#     return torch.cat(probs).numpy()


# raw_seq = ("CTAAATATTAACTGGTCTTGTGAGATGTCTTCTTGGCTGGAGCCTGACCA..."
#            "CAGGAAGAAGAGGCGCCCGGAAAACCTTAGCTCTT")
 
# # Tokenise once with DNABERT‑2’s BPE, then join with spaces
# tokens      = tokenizer.tokenize(raw_seq)
# lime_string = " ".join(tokens)          # e.g. 'C TAA AT ATTA ACT GG TCTT ...'
 
# explainer = LimeTextExplainer(
#     split_expression=r"\s+",            # split on spaces
#     bow=False,
#     class_names=list(model.config.id2label.values())
# )
 
# exp = explainer.explain_instance(
#     lime_string,
#     predict_proba,
#     num_features=15,
#     num_samples=4000                    # ↑ for smoother importance curves
# )
 
# exp.show_in_notebook()                  # or print(exp.as_list())