import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments
import torch

# Set paths
MODEL_PATH = "../indicbart-legal-pretrained"
TRAIN_CSV = "MILDSum_train_2498.csv"
VAL_CSV = "MILDSum_val_312.csv"
TEST_CSV = "MILDSum_test_312.csv"
OUTPUT_DIR = "./indicbart-mildsum-finetuned"

# Step 1: Prepare bilingual summarization dataset
def prepare_mildsum(path):
    df = pd.read_csv(path)
    en_data = pd.DataFrame({
        "input": df["Judgement"].apply(lambda x: f"<2en> {x.strip()} </s>"),
        "summary": df["Eng_summ"]
    })
    hi_data = pd.DataFrame({
        "input": df["Judgement"].apply(lambda x: f"<2hi> {x.strip()} </s>"),
        "summary": df["Hindi_summ"]
    })
    return pd.concat([en_data, hi_data]).reset_index(drop=True)

train_df = prepare_mildsum(TRAIN_CSV)
val_df = prepare_mildsum(VAL_CSV)
test_df = prepare_mildsum(TEST_CSV)

# Step 2: Tokenization
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
max_input_len = 512
max_target_len = 128

def preprocess(example):
    model_input = tokenizer(example["input"], max_length=max_input_len, padding="max_length", truncation=True)
    label = tokenizer(text_target=example["summary"], max_length=max_target_len, padding="max_length", truncation=True)

    model_input["labels"] = label["input_ids"]
    return model_input

train_dataset = Dataset.from_pandas(train_df).map(preprocess, remove_columns=["input", "summary"])
val_dataset = Dataset.from_pandas(val_df).map(preprocess, remove_columns=["input", "summary"])
test_dataset = Dataset.from_pandas(test_df).map(preprocess, remove_columns=["input", "summary"])

# Step 3: Load model
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)

# Step 4: Setup training
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=100,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=5,
    learning_rate=3e-5,
    warmup_steps=200,
    weight_decay=0.01,
    save_total_limit=2,
    predict_with_generate=True,
    fp16=torch.cuda.is_available(),
    push_to_hub=False
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# Step 5: Train
trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# Step 6: Evaluate on test set
metrics = trainer.evaluate(eval_dataset=test_dataset)
print("Final Evaluation Metrics on Test Set:")
print(metrics)
