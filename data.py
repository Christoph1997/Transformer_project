import numpy as np

from datasets import load_dataset
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
from collections import Counter


class Dataset:
    def __init__(self, model_name):
        self.model_name = model_name
        self.dataset = None
        self.tokenizer = None

    def load_dataset(self, dataset_name):
        # Load dataset
        self.dataset = load_dataset(dataset_name)

        # Subsample for faster training
        self.dataset["train"] = self.dataset["train"].shuffle(seed=42).select(range(10000))
        self.dataset["test"] = self.dataset["test"].shuffle(seed=42).select(range(2000))

    def load_tokenizer(self):
        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def preprocess(self):
        # Preprocessing function
        def tokenize_dataset(example):
            texts = [t + " " + c for t, c in zip(example["title"], example["content"])]
            return self.tokenizer(texts, padding="max_length", truncation=True, max_length=256)

        # Tokenize dataset
        # Stick with batched=True for faster processing
        tokenized_dataset = self.dataset.map(tokenize_dataset, batched=True)
        tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
        tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

        return tokenized_dataset
    
    def analyze_dataset(self, result_path):
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
        