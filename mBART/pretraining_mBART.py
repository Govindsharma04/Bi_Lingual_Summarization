import pandas as pd
from datasets import load_dataset
from langdetect import detect
from transformers import MBart50Tokenizer, MBartForConditionalGeneration, DataCollatorForLanguageModeling, TrainingArguments, Trainer

# Example using Hugging Face Dataset
dataset = load_dataset("csv", data_files={"data": "the_dataset.csv"})["data"]

dataset = dataset.train_test_split(test_size=0.02, seed=42)
train_dataset = dataset["train"]
val_dataset = dataset["test"]

len(val_dataset.to_pandas())

# Load the tokenizer
tokenizer = MBart50Tokenizer.from_pretrained("facebook/mbart-large-50", src_lang="hi_IN")

# Function to detect language
def detect_language(text):
    lang = detect(text)
    return "hi_IN" if lang == "hi" else "en_XX"

# Tokenization function with dynamic src_lang
def tokenize_function(examples):
    # Detect language and set src_lang dynamically
    src_langs = [detect_language(text) for text in examples["Text"]]
    tokenized_texts = []
    for text, src_lang in zip(examples["Text"], src_langs):
        tokenizer.src_lang = src_lang
        tokenized_texts.append(tokenizer(text, padding="max_length", truncation=True, max_length=512))
    
    # Combine the tokenized outputs into a dictionary
    return {
        "input_ids": [tokenized["input_ids"] for tokenized in tokenized_texts],
        "attention_mask": [tokenized["attention_mask"] for tokenized in tokenized_texts],
    }

# Tokenize the datasets
tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_val = val_dataset.map(tokenize_function, batched=True)

# Data collator for language modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

# Load the model
model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50")

# Training arguments
training_args = TrainingArguments(
    output_dir="./mbart-pretrained-legal",
    per_device_train_batch_size=8,               # Adjust based on GPU memory
    per_device_eval_batch_size=8,
    evaluation_strategy="steps",
    eval_steps=2000,
    save_steps=2000,
    save_total_limit=3,
    learning_rate=3e-5,
    num_train_epochs=3,                          # Try increasing based on convergence
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=500,
    save_strategy="steps",
    fp16=True,                                    # Enables mixed precision (if your GPU supports it)
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# Train the model
trainer.train()

# Save the model
trainer.save_model("./mbart-pretrained-legal")
# Save the tokenizer
tokenizer.save_pretrained("./mbart-pretrained-legal")
# Save the dataset
train_dataset.save_to_disk("./mbart-pretrained-legal/train_dataset")
val_dataset.save_to_disk("./mbart-pretrained-legal/val_dataset")