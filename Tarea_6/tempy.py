from collections import defaultdict, Counter
import json

from matplotlib import pyplot as plt
import numpy as np
import torch

from transformers import AutoModelForSequenceClassification, DistilBertForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-cased', num_labels=2)

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased") # convenient! Defaults to Fast
print(tokenizer)

input_str = "I dub thee the unforgiven"


model_inputs = tokenizer(input_str, return_tensors="pt")

# Option 1
model_outputs = model(input_ids=model_inputs.input_ids, attention_mask=model_inputs.attention_mask)

# Option 2 - the keys of the dictionary the tokenizer returns are the same as the keyword arguments
#            the model expects

# f({k1: v1, k2: v2}) = f(k1=v1, k2=v2)

model_outputs = model(**model_inputs)



print(model_inputs)
print()
print(model_outputs)
print()
print(f"Distribution over labels: {torch.softmax(model_outputs.logits, dim=1)}")

from transformers import AutoModel

model = AutoModel.from_pretrained("distilbert-base-cased", output_attentions=True, output_hidden_states=True)
model.eval()

model_inputs = tokenizer(input_str, return_tensors="pt")
with torch.no_grad():
    model_output = model(**model_inputs)


print("Hidden state size (per layer):  ", model_output.hidden_states[0].shape)
print("Attention head size (per layer):", model_output.attentions[0].shape)     # (layer, batch, query_word_idx, key_word_idxs)
                                                                               # y-axis is query, x-axis is key
print(model_output)


import os
import random
from transformers import AutoTokenizer
from datasets import Dataset, DatasetDict

# Function to create one-hot encoding for authors
def encode_author(author, authors_list):
    encoding = [1 if a == author else 0 for a in authors_list]
    return encoding

# Function to split text into chunks of equal length
def split_into_chunks(text, chunk_size):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# Define authors and paths
authors = os.listdir("book_datasets")
data = {'text': [], 'label': []}

# Iterate over authors and text files
for author in authors:
    author_path = os.path.join("book_datasets", author)
    author_files = [f for f in os.listdir(author_path) if f.endswith(".txt")]

    for file in author_files:
        file_path = os.path.join(author_path, file)
        
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
            
            # Split text into chunks of 512 tokens
            chunks = split_into_chunks(text, 512)
            
            # Create one-hot encoding for the author
            label = encode_author(author, authors)
            
            data['text'].extend(chunks)
            data['label'].extend([label] * len(chunks))

# Shuffle the dataset
random.shuffle(data['text'])
random.shuffle(data['label'])

# Split the dataset into training and validation sets
train_size = int(0.8 * len(data['text']))
train_data = {'text': data['text'][:train_size], 'label': data['label'][:train_size]}
val_data = {'text': data['text'][train_size:], 'label': data['label'][train_size:]}

# Create datasets
book_train_dataset = Dataset.from_dict(train_data)
book_val_dataset = Dataset.from_dict(val_data)

# Create a DatasetDict
book_dataset = DatasetDict({'train': book_train_dataset, 'val': book_val_dataset})

# Tokenize dataset
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")
tokenized_dataset = book_dataset.map(lambda example: tokenizer(example['text'], truncation=True), batched=True)

tokenized_dataset = tokenized_dataset.remove_columns(["text"])
tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
tokenized_dataset.set_format("torch")


### TRAINING

from transformers import TrainingArguments, Trainer

model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-cased', num_labels=3)

arguments = TrainingArguments(
    output_dir="sample_hf_trainer",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    evaluation_strategy="epoch", # run validation at the end of each epoch
    save_strategy="epoch",
    learning_rate=2e-5,
    load_best_model_at_end=True,
    seed=224
)


def compute_metrics(eval_pred):
    """Called at the end of validation. Gives accuracy"""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    # calculates the accuracy
    return {"accuracy": np.mean(predictions == labels)}


trainer = Trainer(
    model=model,
    args=arguments,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['val'],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

from transformers import TrainerCallback, EarlyStoppingCallback

class LoggingCallback(TrainerCallback):
    def __init__(self, log_path):
        self.log_path = log_path

    def on_log(self, args, state, control, logs=None, **kwargs):
        _ = logs.pop("total_flos", None)
        if state.is_local_process_zero:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(logs) + "\n")


trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=1, early_stopping_threshold=0.0))
trainer.add_callback(LoggingCallback("sample_hf_trainer/log.jsonl"))

trainer.train()