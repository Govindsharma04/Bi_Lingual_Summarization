import pandas as pd
from datasets import Dataset, concatenate_datasets
from transformers import (
    MBart50TokenizerFast,
    MBartForConditionalGeneration,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer
)
import torch

# Load CSVs
train_df = pd.read_csv("MILDSum_train_2185.csv")
val_df = pd.read_csv("MILDSum_val_469.csv")
test_df = pd.read_csv("MILDSum_test_468.csv")

# Duplicate rows: one for Hindi summary, one for English summary
def prepare_multilingual_df(df):
    en_data = df[["Judgement", "Eng_summ"]].rename(columns={"Judgement": "text", "Eng_summ": "summary"})
    en_data["lang"] = "en_XX"

    hi_data = df[["Judgement", "Hindi_summ"]].rename(columns={"Judgement": "text", "Hindi_summ": "summary"})
    hi_data["lang"] = "hi_IN"

    return pd.concat([en_data, hi_data], ignore_index=True)

train_df = prepare_multilingual_df(train_df)
val_df = prepare_multilingual_df(val_df)

# Convert to Huggingface Datasets
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# Load tokenizer and model
tokenizer = MBart50TokenizerFast.from_pretrained("./mbart-pretrained-legal")  # your pretrained path
model = MBartForConditionalGeneration.from_pretrained("./mbart-pretrained-legal")

# Preprocessing function
def preprocess_function(examples):
    tokenizer.src_lang = examples["lang"]
    inputs = tokenizer(
        examples["text"],
        max_length=512,
        padding="max_length",
        truncation=True
    )
    targets = tokenizer(
        examples["summary"],
        max_length=128,
        padding="max_length",
        truncation=True
    )

    inputs["labels"] = targets["input_ids"]

    inputs["forced_bos_token_id"] = tokenizer.lang_code_to_id[examples["lang"]]
    return inputs

# Apply preprocessing
train_dataset = train_dataset.map(preprocess_function, remove_columns=train_dataset.column_names)
val_dataset = val_dataset.map(preprocess_function, remove_columns=val_dataset.column_names)

# Data collator
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# Training arguments
training_args = TrainingArguments(
    output_dir="./mbart-finetuned-mildsum",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=2,
    learning_rate=3e-5,
    num_train_epochs=3,
    evaluation_strategy="steps",
    eval_steps=1000,
    save_steps=1000,
    save_total_limit=2,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
    save_strategy="steps",
    fp16=torch.cuda.is_available(),
    report_to="none"
)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# Train
trainer.train()

# Save
trainer.save_model("./mbart-finetuned-mildsum")
tokenizer.save_pretrained("./mbart-finetuned-mildsum")

