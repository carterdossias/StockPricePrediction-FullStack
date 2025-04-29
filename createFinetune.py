import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

# Convert CSV into Pandas Data frame
df = pd.read_csv("labeledNews.csv")  # columns: "Sentence", "Sentiment"

# convert pandas dataframe to hugging face dataset
dataset = Dataset.from_pandas(df)

#splitting the data randomly into training and testing (currently 80% train and 20% test)
train_test = dataset.train_test_split(test_size=0.2)
dataset_train = train_test["train"]
dataset_val = train_test["test"]

# cleaning and mapping the data
def cleanData(example):
    example["text"] = example["Sentence"]
    return example

# mapping the cleanData function to the dataset
dataset_train = dataset_train.map(cleanData)
dataset_val = dataset_val.map(cleanData)

labelToID = {"negative": 0, "neutral": 1, "positive": 2}

def encode_labels(example):
    example["label"] = labelToID[example["Sentiment"].lower()]
    return example

dataset_train = dataset_train.map(encode_labels)
dataset_val = dataset_val.map(encode_labels)

# tokenize the data
model_checkpoint = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

def tokenize_function(example):
    return tokenizer(example["text"], padding="max_length", truncation=True, max_length=512)

dataset_train = dataset_train.map(tokenize_function, batched=True)
dataset_val = dataset_val.map(tokenize_function, batched=True)

# loading up the model
model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint,
    num_labels=3,
    id2label={0: "negative", 1: "neutral", 2: "positive"},
    label2id={"negative": 0, "neutral": 1, "positive": 2},
    problem_type="single_label_classification"
)

# 7. Set up training args & Trainer
training_args = TrainingArguments(
    output_dir="./finbert-finetuned",
    do_train=True,
    do_eval=True,
    # run evaluation every N steps (e.g. every epoch worth of steps)
    eval_steps= len(dataset_train) // 16,
    logging_dir='./logs',
    logging_steps=50,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_train,
    eval_dataset=dataset_val,
    tokenizer=tokenizer,
)

# 8. Train, save, evaluate
trainer.train()
model.save_pretrained("./finbert-finetuned")
tokenizer.save_pretrained("./finbert-finetuned")

results = trainer.evaluate()
print(results)