import pandas as pd
import re
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load fine-tuned model and tokenizer
model_dir = "legal-pegasus-mildsum-final_2"
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to("cuda")

# Prefixes for bilingual summary prompts
prefixes = {
    'Hindi': "न्यायालय के फैसले का सारांश हिंदी में दें: ",
    'English': "Summarize this legal judgment in English: "
}

# Tokenization limits
max_input_length = 1024
max_target_length = 256

# Load test data
df = pd.read_csv("MILDSum_test_468.csv")

# Clean judgments
def clean_text(text):
    return re.sub(r'\[\d+\]', '', str(text))

df['Judgement'] = df['Judgement'].apply(clean_text)

# Initialize storage for results
hindi_rows = []
english_rows = []

# Inference function
def generate_summary(text, lang):
    input_text = prefixes[lang] + text
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_length,
        padding="max_length"
    ).to("cuda")

    summary_ids = model.generate(
        inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_length=max_target_length,
        num_beams=4,
        early_stopping=True
    )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Generate summaries
for _, row in df.iterrows():
    judgment = row['Judgement']

    # Hindi
    ref_hi = row['Hindi_summ']
    pred_hi = generate_summary(judgment, "Hindi")
    hindi_rows.append({
        "Judgement": judgment,
        "Hindi_summ": ref_hi,
        "generated_hindi_summ": pred_hi
    })

    # English
    ref_en = row['English_summ']
    pred_en = generate_summary(judgment, "English")
    english_rows.append({
        "Judgement": judgment,
        "Eng_summ": ref_en,
        "generated_eng_summ": pred_en
    })

# Save output CSVs
pd.DataFrame(hindi_rows).to_csv("pegasus_hindi_predictions.csv", index=False)
pd.DataFrame(english_rows).to_csv("pegasus_english_predictions.csv", index=False)

print("✅ Summaries generated and saved.")
