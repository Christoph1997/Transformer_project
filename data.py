from datasets import load_dataset
from transformers import BertTokenizerFast


class Dataset:
    def __init__(self, model_name):
        self.model_name = model_name
        self.dataset = None
        self.tokenizer = None

    def load_dataset(self):
        # Load dataset (Amazon Polarity)
        self.dataset = load_dataset("amazon_polarity")

        # Subsample for faster training
        self.dataset["train"] = self.dataset["train"].shuffle(seed=42).select(range(10000))
        self.dataset["test"] = self.dataset["test"].select(range(2000))

    def load_tokenizer(self):
        # Tokenizer
        self.tokenizer = BertTokenizerFast.from_pretrained(self.model_name)

    def preprocess(self):
        # 4. Preprocessing function
        def tokenize_dataset(example):
            text = example["title"] + " " + example["content"]
            return self.tokenizer(text, padding="max_length", truncation=True, max_length=256)

        # 5. Tokenize dataset
        tokenized_dataset = self.dataset.map(tokenize_dataset, batched=True)
        tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
        tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

        return tokenized_dataset
