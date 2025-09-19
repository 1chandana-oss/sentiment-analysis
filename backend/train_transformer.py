import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

#Load dataset

data_path = os.path.join("data", "IMDB Dataset.csv")
df = pd.read_csv(data_path)


# Add neutral class

def assign_label(sentiment, text):
    if len(text.split()) < 5:
        return "neutral"
    return sentiment.lower()  # positive/negative

df["label"] = df.apply(lambda row: assign_label(row["sentiment"], row["review"]), axis=1)


# Train/test split

train_texts, test_texts, train_labels, test_labels = train_test_split(
    df["review"], df["label"], test_size=0.2, random_state=42
)

# Convert to HuggingFace Dataset

train_dataset = Dataset.from_dict({
    "text": train_texts.tolist(),
    "label": train_labels.tolist()
})
test_dataset = Dataset.from_dict({
    "text": test_texts.tolist(),
    "label": test_labels.tolist()
})


# Encode labels
le = LabelEncoder()
le.fit(["negative", "neutral", "positive"])

def encode_label(example):
    example["label"] = le.transform([example["label"]])[0]
    return example

train_dataset = train_dataset.map(encode_label)
test_dataset = test_dataset.map(encode_label)


# Tokenize dataset

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True, max_length=128)

train_dataset = train_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)


# Load model

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)


# Training arguments 
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    logging_dir="./logs",
    save_total_limit=1
)


# Trainer setup

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer
)


# Train model

trainer.train()

#Save fine-tuned model
save_dir = "sentiment_transformer_model"
trainer.save_model(save_dir)
tokenizer.save_pretrained(save_dir)

print(f"✅ Fine-tuned model saved at '{save_dir}/'")
