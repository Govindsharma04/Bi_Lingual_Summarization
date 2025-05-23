import pandas as pd
from datasets import load_dataset
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
import evaluate
import torch

# Load the test dataset
dataset = load_dataset("csv", data_files={"test": "MILDSum_test_468.csv"})["test"]

# Load finetuned model and tokenizer
model_path = "../mbart-finetuned-mildsum"
model = MBartForConditionalGeneration.from_pretrained(model_path)
tokenizer = MBart50TokenizerFast.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define source and target language
SRC_LANG = "en_XX"
TGT_LANG = "hi_IN"  # Or "en_XX" if you're evaluating English summaries

tokenizer.src_lang = SRC_LANG
forced_bos_token_id = tokenizer.lang_code_to_id[TGT_LANG]

# Prepare evaluation metrics
rouge = evaluate.load("rouge")
bleu = evaluate.load("sacrebleu")

# Generate summaries
references = []
predictions = []

for example in dataset:
    input_text = example["Judgement"]
    target_summary = example["Hindi_summ"]  # or "Eng_Summary"

    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding="max_length", max_length=512).to(device)

    output_ids = model.generate(
        **inputs,
        max_length=128,
        num_beams=4,
        forced_bos_token_id=forced_bos_token_id
    )

    decoded_pred = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    predictions.append(decoded_pred)
    references.append(target_summary)

# Evaluate
rouge_result = rouge.compute(predictions=predictions, references=references)
bleu_result = bleu.compute(predictions=predictions, references=[[ref] for ref in references])

print("\n📊 Evaluation Results:")
print("ROUGE:", rouge_result)
print("BLEU:", bleu_result)
