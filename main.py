"""
This is the main script for the BERT fine-tuning project.
It initializes the dataset, tokenizer, and model, and then trains the model on the dataset.
"""


import time
import torch

import data
import models
import trainer


def main():
    
    # Measure time
    start_time = time.perf_counter()

    model_name = "bert-base-uncased"
    epochs = 2
    batch_size = 16

    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset and tokenize it
    print("Initialize dataset and tokenizer.")
    dataset = data.Dataset(model_name)
    dataset.load_dataset()
    dataset.load_tokenizer()
    tokenized_dataset = dataset.preprocess()
    print("Dataset and tokenizer initialized.")

    # Initialize model
    print("Initialize model.")
    model = models.Model(model_name)
    model.load_model()
    print("Model initialized.")

    # Initialize trainer and finetune model
    print("Initialize training arguments.")
    trainer_instance = trainer.Trainer(model.model, tokenized_dataset)
    trainer_instance.set_training_arguments(epochs=epochs, batch_size=batch_size)  
    trainer_instance.train()
    trainer_instance.evaluate()
    print("Model trained and evaluated.")

    # Measure time
    end_time = time.perf_counter()
    print(f"Total time: {end_time - start_time:.4f} seconds.")


if __name__ == "__main__":
    main()