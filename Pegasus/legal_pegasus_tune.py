# -*- coding: utf-8 -*-
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import Dataset, load_metric
import pandas as pd
import numpy as np
import torch
import re

# 1. Dataset Preparation ------------------------------------------------------
def load_and_preprocess(csv_paths):
    dfs = {split: pd.read_csv(path) for split, path in csv_paths.items()}
    
    # Add bilingual prefixes and clean text
    legal_prefixes = {
        'Hindi': "न्यायालय के फैसले का सारांश हिंदी में दें: ",  # Hindi prefix
        'English': "Summarize this legal judgment in English: "
    }

    def process_split(df):
        rows = []
        for _, row in df.iterrows():
            # Clean legal text
            judgment = re.sub(r'\[\d+\]', '', row['Judgement'])  # Remove citations
            for lang in ['Hindi', 'English']:
                rows.append({
                    'input_text': f"{legal_prefixes[lang]}{judgment}",
                    'target_text': row[f'{lang}_summ'].strip(),
                    'language': lang
                })
        return pd.DataFrame(rows).sample(frac=1)  # Shuffle

    return {k: process_split(v) for k, v in dfs.items()}

# 2. Tokenization -------------------------------------------------------------
model_name = "nsi319/legal-pegasus"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Configure lengths based on paper statistics :cite[3]
#max_input_length = 6144  # 6,200 tokens + buffer
#max_target_length = 1024  # 950 tokens + buffer
max_input_length = 1024  # Matches model.config.max_position_embeddings
max_target_length = 256  # Adjust based on decoder's max

def tokenize_function(examples):
    model_inputs = tokenizer(
        examples["input_text"],
        max_length=max_input_length,
        padding="max_length",
        truncation=True,
    )

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples["target_text"],
            max_length=max_target_length,
            padding="max_length",
            truncation=True,
        )

    # Replace padding token id's in labels by -100 so they are ignored in loss
    labels_ids = labels["input_ids"]
    labels_ids = [
        [(label if label != tokenizer.pad_token_id else -100) for label in example]
        for example in labels_ids
    ]

    model_inputs["labels"] = labels_ids
    return model_inputs



# 3. Training Setup -----------------------------------------------------------
# Optimized for NVIDIA L40S 46GB GPU
training_args = Seq2SeqTrainingArguments(
    output_dir="./legal-pegasus-mildsum_2",
    evaluation_strategy="steps",
    eval_steps=500,
    save_steps=500,
    learning_rate=5e-5,
    per_device_train_batch_size=16,  # Maximize GPU utilization
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,
    weight_decay=0.01,
    num_train_epochs=5,
    fp16=True,
    bf16=False,  # Disable for legal text stability
    logging_dir='./logs',
    logging_steps=100,
    save_total_limit=2,
    predict_with_generate=True,
    generation_max_length=max_target_length,
    generation_num_beams=4,
    warmup_steps=500,
    gradient_checkpointing=True,  # Reduce memory usage
    report_to="tensorboard",
    optim="adafactor"
)

# 4. Metric Calculation -------------------------------------------------------
rouge = load_metric("rouge")

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    
    # Replace -100 in labels
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Legal-specific ROUGE calculation
    result = rouge.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        use_stemmer=True,
        use_aggregator=True
    )
    
    return {k: round(v.mid.fmeasure * 100, 2) for k, v in result.items()}

# 5. Execution Pipeline -------------------------------------------------------
def main():
    # Load data
    csv_paths = {
        'train': './MILDSum_train_2185.csv',
        'val': './MILDSum_val_469.csv',
        'test': './MILDSum_test_468.csv'
    }
    datasets = load_and_preprocess(csv_paths)
    
    # Convert to HF Dataset
    hf_datasets = {k: Dataset.from_pandas(v) for k, v in datasets.items()}
    
    # Tokenize
    tokenized_datasets = {
        k: v.map(tokenize_function, batched=True, remove_columns=['input_text', 'target_text'])
        for k, v in hf_datasets.items()
    }
    
    # Initialize Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['val'],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Save final model
    trainer.save_model("./legal-pegasus-mildsum-final_2")
    tokenizer.save_pretrained("./legal-pegasus-mildsum-final_2")
    
    # Evaluate
    test_results = trainer.evaluate(tokenized_datasets['test'])
    print(f"Final Test Results: {test_results}")

if __name__ == "__main__":
    main()
