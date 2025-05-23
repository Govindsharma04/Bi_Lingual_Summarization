import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import numpy as np
import nltk

nltk.download('punkt')  # Only needed once

# Load data
df = pd.read_csv("generated_summaries.csv")

# Ensure there are no NaNs
df.fillna("", inplace=True)

# Extract relevant columns
en_refs = df["Eng_summ"].astype(str).tolist()
en_preds = df["Generated_English_Summary"].astype(str).tolist()

hi_refs = df["Hindi_summ"].astype(str).tolist()
hi_preds = df["Generated_Hindi_Summary"].astype(str).tolist()

# Initialize
smoothie = SmoothingFunction().method4
rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

def compute_bleu(preds, refs):
    scores = []
    for pred, ref in zip(preds, refs):
        score = sentence_bleu([ref.split()], pred.split(), smoothing_function=smoothie)
        scores.append(score)
    return np.mean(scores)

def compute_rouge(preds, refs):
    r1, r2, rl = [], [], []
    for pred, ref in zip(preds, refs):
        scores = rouge.score(ref, pred)
        r1.append(scores['rouge1'].fmeasure)
        r2.append(scores['rouge2'].fmeasure)
        rl.append(scores['rougeL'].fmeasure)
    return {
        "ROUGE-1": np.mean(r1),
        "ROUGE-2": np.mean(r2),
        "ROUGE-L": np.mean(rl)
    }

# 🔍 English Evaluation
print("\n📘 English Summary Evaluation:")
bleu_en = compute_bleu(en_preds, en_refs)
rouge_en = compute_rouge(en_preds, en_refs)
print(f"BLEU: {bleu_en:.4f}")
for k, v in rouge_en.items():
    print(f"{k}: {v:.4f}")

# 🔍 Hindi Evaluation
print("\n🟠 Hindi Summary Evaluation:")
bleu_hi = compute_bleu(hi_preds, hi_refs)
rouge_hi = compute_rouge(hi_preds, hi_refs)
print(f"BLEU: {bleu_hi:.4f}")
for k, v in rouge_hi.items():
    print(f"{k}: {v:.4f}")
