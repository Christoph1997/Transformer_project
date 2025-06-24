import numpy as np

from datasets import load_dataset, concatenate_datasets, Features, Value
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
from collections import Counter


class Dataset:
    def __init__(self, model_name):
        self.model_name = model_name
        self.dataset = {}
        self.tokenized_dataset = {}
        self.tokenizer = None

    def load_dataset(self, dataset_name, train_samples):
        # Load dataset
        if dataset_name == "McAuley-Lab/Amazon-Reviews-2023":
            # Load Amazon Reviews Multi dataset
            self.dataset = load_dataset(dataset_name, "raw_review_All_Beauty", trust_remote_code=True)
            # Shift ratings from 1–5 to 0–4
            self.dataset["full"] = self.dataset["full"].map(lambda x: {"rating": x["rating"] - 1})
            # Shuffle and create manual splits
            self.dataset = self.dataset.shuffle(seed=42)
            # Cast label to int
            self.dataset["full"] = self.dataset["full"].cast(Features({**self.dataset["full"].features,"rating": Value("int32")}))
            split = self.dataset["full"].train_test_split(test_size=int(train_samples/5), seed=42)
            # Get 2000 examples per class (for a total of 10,000 with 5 classes)
            balanced_train_splits = []
            for label in range(5):
                per_class = split["train"].filter(lambda x: x["rating"] == label)
                balanced_per_class = per_class.shuffle(seed=42).select(range(int(train_samples/5)))
                balanced_train_splits.append(balanced_per_class)
            self.dataset["train"] = concatenate_datasets(balanced_train_splits)
            self.dataset["test"] = split["test"]
            self.dataset = self.dataset.rename_column("rating", "label")
        else:
            self.dataset = load_dataset(dataset_name)
            # Subsample for faster training
            self.dataset["train"] = self.dataset["train"].shuffle(seed=42).select(range(train_samples))
            self.dataset["test"] = self.dataset["test"].shuffle(seed=42).select(range(int(train_samples/5)))

    def load_tokenizer(self):
        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def preprocess(self, dataset_name):
        # Preprocessing function
        if dataset_name == "McAuley-Lab/Amazon-Reviews-2023":
            def tokenize_dataset(example):
                texts = [t + " " + c for t, c in zip(example["title"], example["text"])]
                return self.tokenizer(texts, padding="max_length", truncation=True, max_length=256)
        else:
            def tokenize_dataset(example):
                texts = [t + " " + c for t, c in zip(example["title"], example["content"])]
                return self.tokenizer(texts, padding="max_length", truncation=True, max_length=256)

        # Tokenize dataset
        # Stick with batched=True for faster processing
        tokenized_dataset = self.dataset.map(tokenize_dataset, batched=True)
        tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
        tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
        self.tokenized_dataset = tokenized_dataset
    
    def analyze_dataset(self, result_path, dataset_name):
        """Analyze dataset"""
        print("Dataset analysis:")
        print(f"Number of training examples: {len(self.dataset['train'])}")
        print(f"Number of test examples: {len(self.dataset['test'])}")
        print(f"Features: {self.dataset['train'].features}")
        print(f"Sample training example: {self.dataset['train'][0]}")
        print(f"Sample test example: {self.dataset['test'][0]}")

        # Plot train label distribution
        train_counts = Counter(self.dataset['train']["label"])
        test_counts = Counter(self.dataset['test']["label"])
        if dataset_name == "McAuley-Lab/Amazon-Reviews-2023":
            # For Amazon Reviews Multi, labels are 1-5
            label_names = [f"{i} Star" for i in range(1,6)]
        else:
            label_names = self.dataset['train'].features['label'].names

        # Ensure order is consistent
        labels = label_names
        x = np.arange(len(labels))
        train_values = [train_counts[i] for i in range(len(labels))]
        test_values = [test_counts[i] for i in range(len(labels))]

        # Plot
        width = 0.35
        plt.figure(figsize=(8, 5))
        plt.bar(x - width/2, train_values, width, label='Train', color='skyblue')
        plt.bar(x + width/2, test_values, width, label='Test', color='salmon')

        plt.xticks(x, labels)
        plt.xlabel("Label")
        plt.ylabel("Count")
        plt.title("Label Distribution in Train and Test Sets")
        plt.legend()
        plt.grid(axis='y')
        plt.savefig(f"{result_path}/label_distribution_train_test.png")
        plt.close()
        