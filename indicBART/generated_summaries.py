from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import Dataset
import pandas as pd
import torch
from tqdm import tqdm

# Paths
MODEL_PATH = "./indicbart-mildsum-finetuned"
TEST_CSV = "MILDSum_test_312.csv"
OUTPUT_CSV = "generated_summaries.csv"

# Load model & tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH).to("cuda" if torch.cuda.is_available() else "cpu")
# Load and prepare bilingual input
df = pd.read_csv(TEST_CSV)

en_inputs = df["Judgement"].apply(lambda x: f"<2en> {x.strip()} </s>")
hi_inputs = df["Judgement"].apply(lambda x: f"<2hi> {x.strip()} </s>")

def generate_summary(texts):
    inputs = tokenizer(texts.tolist(), return_tensors="pt", max_length=512, truncation=True, padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=128,
            num_beams=4,
            early_stopping=True
        )
    
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)

# Generate in batches
batch_size = 16
en_summaries, hi_summaries = [], []

for i in tqdm(range(0, len(df), batch_size)):
    en_batch = en_inputs[i:i+batch_size]
    hi_batch = hi_inputs[i:i+batch_size]
    
    en_summaries += generate_summary(en_batch)
    hi_summaries += generate_summary(hi_batch)

df["Generated_English_Summary"] = en_summaries
df["Generated_Hindi_Summary"] = hi_summaries

df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved to {OUTPUT_CSV}")
