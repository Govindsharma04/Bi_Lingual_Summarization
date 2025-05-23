import pandas as pd
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from tqdm import tqdm
import torch

# Load the fine-tuned model and tokenizer
model_dir = "../mbart-finetuned-mildsum"
tokenizer = MBart50TokenizerFast.from_pretrained(model_dir)
model = MBartForConditionalGeneration.from_pretrained(model_dir).to("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
df = pd.read_csv("MILDSum_test_468.csv")
df = df.dropna(subset=["Judgement", "Eng_summ", "Hindi_summ"])

source_texts = df["Judgement"].tolist()

generated_eng_summ = []
generated_hindi_summ = []

print("Generating summaries using fine-tuned mBART...")

# Generate English and Hindi summaries
for text in tqdm(source_texts):
    # English summary generation
    tokenizer.src_lang = "en_XX"
    encoded_en = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=1024).to(device)
    gen_en = model.generate(**encoded_en, forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"], max_length=256, num_beams=4)
    summary_en = tokenizer.decode(gen_en[0], skip_special_tokens=True)
    generated_eng_summ.append(summary_en)

    # Hindi summary generation
    tokenizer.src_lang = "hi_IN"
    encoded_hi = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=1024).to(device)
    gen_hi = model.generate(**encoded_hi, forced_bos_token_id=tokenizer.lang_code_to_id["hi_IN"], max_length=256, num_beams=4)
    summary_hi = tokenizer.decode(gen_hi[0], skip_special_tokens=True)
    generated_hindi_summ.append(summary_hi)

# Add generated summaries to DataFrame
df["Generated_eng_summ"] = generated_eng_summ
df["Generated_hindi_summ"] = generated_hindi_summ

# Save to CSV
df[["Judgement", "Eng_summ", "Hindi_summ", "Generated_eng_summ", "Generated_hindi_summ"]].to_csv("mbart_generated_summaries.csv", index=False)

print("✅ Saved to mbart_generated_summaries.csv")
