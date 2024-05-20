import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AlbertForSequenceClassification, AlbertTokenizer, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import ast

def drop_unused_columns(df: pd.DataFrame):
    columns = 'courseItemsNumber,id,numberOfArticles,readingTime,updatedAt,numberOfTopics,createdAt,version'
    for column in columns.split(','):
        if column in df.columns:
            del df[column]

def remove_nan_from_array(array: list):
    return [data for data in array if not isinstance(data, float)]

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    logits = torch.tensor(logits) 
    labels = torch.tensor(labels)  
    predictions = torch.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels.numpy(), predictions.numpy(), average='weighted')
    acc = accuracy_score(labels.numpy(), predictions.numpy())
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

df = pd.read_csv('./data.csv')
drop_unused_columns(df)

print("Number of courses: ", len(df))
print("Columns: ", list(df.columns))

for column in df.columns:
    df[column] = df[column].fillna('').astype(str)

df['combined_text'] = df.apply(lambda row: ' '.join(row.values), axis=1)

def safe_literal_eval(val):
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return []

df['categories'] = df['categories'].apply(lambda x: safe_literal_eval(x) if isinstance(x, str) else [])

all_categories = sorted(set(cat for sublist in df['categories'] for cat in sublist))
category_to_id = {category: idx for idx, category in enumerate(all_categories)}

df['category_ids'] = df['categories'].apply(lambda x: [category_to_id[cat] for cat in x])

df = df[df['category_ids'].apply(len) > 0]

texts = df['combined_text'].tolist()
labels = df['category_ids'].tolist()

train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

train_labels = [label[0] for label in train_labels if len(label) > 0]
test_labels = [label[0] for label in test_labels if len(label) > 0]

train_dataset = Dataset.from_dict({'text': train_texts, 'label': train_labels})
test_dataset = Dataset.from_dict({'text': test_texts, 'label': test_labels})
datasets = DatasetDict({'train': train_dataset, 'test': test_dataset})

model_name = "albert-base-v2"
tokenizer = AlbertTokenizer.from_pretrained(model_name)
model = AlbertForSequenceClassification.from_pretrained(model_name, num_labels=len(category_to_id))

def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True)

tokenized_datasets = datasets.map(tokenize_function, batched=True)

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8, 
    per_device_eval_batch_size=8, 
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test'],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()

eval_results = trainer.evaluate()

print(eval_results)

model.save_pretrained('./saved_model')
tokenizer.save_pretrained('./saved_model')
