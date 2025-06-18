"""
This is the main script for the BERT fine-tuning project.
It initializes the dataset, tokenizer, and model, and then trains the model on the dataset.
"""


import time
import torch
import csv
import os
import matplotlib.pyplot as plt

import data
import models
import trainer


def main():

    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == torch.device("cuda"):
         torch.cuda.empty_cache()  # Clear GPU memory if using CUDA
    print(f"Using device: {device}")

    # Parameters for all models
    epochs = 2
    batch_size = 16  # Default batch size, will be adjusted for Longformer
    dataset_name = "amazon_polarity"

    # List of model names to iterate over
    model_names = [
        "bert-base-cased",
        "roberta-base",
        "distilbert-base-cased",
        "xlnet-base-cased",
        "google/electra-base-discriminator",
        "facebook/bart-base",
    ]

    # Iterate over models
    results_dict = {}
    for model_name in model_names:
        print(f"\n\nCalculating {model_name} model with {epochs} epochs and batch size {batch_size}.")
        results_dict[model_name] = run_model(model_name, epochs, batch_size, dataset_name)

    # Obtain general results
    plot_model_comparison(results_dict)
    print("All models have been trained and evaluated.")


def run_model(model_name, epochs, batch_size, dataset_name):
    """
    Function to run the model fine-tuning and evaluation.
    """
    # Longformer requires a smaller batch size due to memory constraints
    if model_name == "allenai/longformer-base-4096":
         batch_size = 8 
    # Check if result folder exists
    result_path = f"./results/{model_name}_{dataset_name}"
    # Create the folder if it doesn't exist
    os.makedirs(result_path, exist_ok=True)
    
    # Measure time
    start_time = time.perf_counter()

    # Load dataset and tokenize it
    print("Initialize dataset and tokenizer.")
    dataset = data.Dataset(model_name)
    dataset.load_dataset(dataset_name)
    dataset.load_tokenizer()
    tokenized_dataset = dataset.preprocess()
    dataset.analyze_dataset(result_path)
    print("Dataset and tokenizer initialized.")

    # Initialize model
    print("Initialize model.")
    model = models.Model(model_name)
    model.load_model()
    print("Model initialized.")

    # Initialize trainer and finetune model
    print("Initialize training arguments.")
    trainer_instance = trainer.Trainer_instance(model.model, tokenized_dataset)
    trainer_instance.set_training_arguments(epochs=epochs, batch_size=batch_size, dataset_name=dataset_name, model_name=model_name)  
    trainer_instance.train(model_name)
    trainer_instance.visualize_results(model_name, result_path)
    results = trainer_instance.evaluate(model_name)

    print("Model trained and evaluated.")

    # Save model
    print("Saving model.")
    save_path = f"./.models/{model_name}_{dataset_name}"
    trainer_instance.trainer[model_name].save_model(save_path)
    dataset.tokenizer.save_pretrained(save_path)

    # Measure time
    end_time = time.perf_counter()
    total_time = end_time - start_time
    print(f"Total time: {total_time:.4f} seconds.")
    with open(f"{result_path}/time.txt", "w") as file:
        file.write(str(total_time))

    return results


def plot_model_comparison(results_dict):
        """Plot model comparison"""
        # Plotting accuracy and F1 score for each model
        models = list(results_dict.keys())
        accuracies = [results_dict[m]['eval_accuracy'] for m in models]
        f1_scores = [results_dict[m]['eval_f1'] for m in models]

        x = range(len(models))
        plt.figure(figsize=(10, 5))
        plt.bar(x, accuracies, width=0.4, label='Accuracy', align='center')
        plt.bar([i + 0.4 for i in x], f1_scores, width=0.4, label='F1 Score', align='center')
        plt.xticks([i + 0.2 for i in x], models, rotation=45)
        plt.ylabel('Score')
        plt.title('Transformer Model Performance')
        plt.legend()
        plt.tight_layout()
        plt.savefig("./results/accuracy_f1score.png")
        plt.close()
        
        # Save results to CSV
        with open("./results/accuracy_f1score.csv", "w", newline="") as f:
            writer = csv.writer(f)
            # Write header
            writer.writerow(["Model", "Accuracy", "F1 Score"])
            
            # Write each row
            for model, metrics in results_dict.items():
                writer.writerow([model, metrics["eval_accuracy"], metrics["eval_f1"]])


if __name__ == "__main__":
    main()