from datasets import load_dataset
from transformers import AutoTokenizer


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
        self.dataset["test"] = self.dataset["test"].select(range(2000))

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
