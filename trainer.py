import torch
from transformers import TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import csv


class Trainer_instance:
    def __init__(self, model, tokenized_dataset):
        self.model = model
        self.train_dataset = tokenized_dataset["train"]
        self.eval_dataset = tokenized_dataset["test"]
        self.trainer = {}
        self.training_args = None
        self.preds = None

    def set_training_arguments(self, epochs, batch_size, dataset_name, model_name):
        """Training arguments"""
        model_path = f"{model_name}_{dataset_name}"
        self.training_args = TrainingArguments(
            output_dir=f"./results/{model_path}",
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="epoch",
            logging_dir=f"./results/{model_path}",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            save_total_limit=1,
            remove_unused_columns=False,
            report_to="none",  # disables W&B or TensorBoard
        )

    def train(self, model_name, dataset_name):
        """Create Trainer"""
        if model_name == "facebook/bart-base" and dataset_name == "McAuley-Lab/Amazon-Reviews-2023":
            # For BART, we need to use a different compute_metrics function
            trainer = Trainer(
                model=self.model,
                args=self.training_args,
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset,
                compute_metrics=self.compute_multiclass_bart_metrics,
            )
        elif model_name == "facebook/bart-base":
            # For BART, we need to use a different compute_metrics function
            trainer = Trainer(
                model=self.model,
                args=self.training_args,
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset,
                compute_metrics=self.compute_bart_metrics,
            )
        elif dataset_name == "McAuley-Lab/Amazon-Reviews-2023":
            # For multiclass, we need to use a different compute_metrics function
            trainer = Trainer(
                model=self.model,
                args=self.training_args,
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset,
                compute_metrics=self.compute_multiclass_metrics,
            )
        else:
            trainer = Trainer(
                model=self.model,
                args=self.training_args,
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset,
                compute_metrics=self.compute_metrics,
            )
        # Train the model
        trainer.train()
        self.trainer[model_name] = trainer

    @staticmethod
    def compute_metrics(p):
        """Define evaluation metrics"""
        preds = torch.argmax(torch.tensor(p.predictions), axis=1)
        labels = p.label_ids
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds)
        return {"accuracy": acc, "f1": f1}

    @staticmethod
    def compute_bart_metrics(p):
        """Define evaluation metrics for bart"""
        preds = torch.argmax(torch.tensor(p.predictions[0]), axis=1)
        labels = p.label_ids
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds)
        return {"accuracy": acc, "f1": f1}
    
    @staticmethod
    def compute_multiclass_metrics(p):
        """Define evaluation metrics for multiclass classification"""
        preds = torch.argmax(torch.tensor(p.predictions), axis=1)
        labels = p.label_ids
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average="weighted")
        return {"accuracy": acc, "f1": f1}
    
    @staticmethod
    def compute_multiclass_bart_metrics(p):
        """Define evaluation metrics for bart"""
        preds = torch.argmax(torch.tensor(p.predictions[0]), axis=1)
        labels = p.label_ids
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average="weighted")
        return {"accuracy": acc, "f1": f1}

    def evaluate(self, model_name):
        """Evaluate"""
        results = self.trainer[model_name].evaluate()
        print("Evaluation results:", results)
        return results
    
    def visualize_results(self, model_name, result_path, dataset_name, dataset):
        """Visualize results"""
        preds = self.trainer[model_name].predict(self.eval_dataset)
        if model_name == "facebook/bart-base":
            # For BART, we need to use a different prediction
            y_pred = torch.argmax(torch.tensor(preds.predictions[0]), axis=1)
        else:
            # For other models, we can use the standard prediction
            y_pred = torch.argmax(torch.tensor(preds.predictions), axis=1)
        y_true = preds.label_ids
        if dataset_name == "McAuley-Lab/Amazon-Reviews-2023":
            # For Amazon Reviews Multi, labels are 0-4
            label_names = [f"{i} Star" for i in range(1,6)]
        else:
            label_names = self.train_dataset.features["labels"].names

        # Open file to write misclassified samples
        with open(f"{result_path}/misclassified.txt", "w", encoding="utf-8") as f:
            for i, (true, pred) in enumerate(zip(y_true, y_pred)):
                if true != pred:
                    title = dataset["test"][i]["title"]
                    if dataset_name == "McAuley-Lab/Amazon-Reviews-2023":
                        text = dataset["test"][i]["text"]
                    else:
                        text = dataset["test"][i]["content"]
                    error_type = (
                        f"False Positive for class {pred.item()} / "
                        f"False Negative for class {true}"
                    )
                    f.write(f"{error_type}\nTitle: {title}\nText: {text}\nTrue: {true}, Pred: {pred.item()}\n\n")
                
        # Print and save evaluation metrics
        print("Classification Report:")
        report = classification_report(y_true, y_pred, digits=4)
        print(report)
        with open(f"{result_path}/classification_report.txt", "w") as f:
            f.write(report)
            
        print("Confusion Matrix:")
        cm = confusion_matrix(y_true, y_pred)
        print(cm)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
        fig, ax = plt.subplots(figsize=(8, 6))
        disp.plot(ax=ax, cmap='Blues', values_format='d')
        plt.title("Confusion Matrix")
        plt.savefig(f"{result_path}/confusion_matrix.png")
        plt.close()

        # Visualize logging history (accuracy scores)
        if hasattr(self.trainer[model_name], "state"):
            history = self.trainer[model_name].state.log_history
            if history:
                epochs = [log["epoch"] for log in history if "eval_accuracy" in log]
                eval_acc = [log["eval_accuracy"] for log in history if "eval_accuracy" in log]

                plt.figure(figsize=(10, 5))
                plt.plot(epochs, eval_acc, label='Eval Accuracy')
                plt.xlabel('Epochs')
                plt.ylabel('Accuracy')
                plt.title('Training and Evaluation Accuracy Over Epochs')
                plt.legend()
                plt.grid(True)
                plt.savefig(f"{result_path}/accuracy_over_epochs.png")
                plt.close()

                # Write to CSV
                with open(f"{result_path}/eval_accuracy.csv", mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(["Accuracy"])
                    for value in eval_acc:
                        writer.writerow([value])
        
        return cm
