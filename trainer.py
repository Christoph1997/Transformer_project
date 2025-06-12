import torch
from transformers import TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


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
            logging_steps=100,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            save_total_limit=1,
            remove_unused_columns=False,
            report_to="none",  # disables W&B or TensorBoard
        )

    def train(self, model_name):
        """Create Trainer"""
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

    def evaluate(self, model_name):
        """Evaluate"""
        results = self.trainer[model_name].evaluate()
        print("Evaluation results:", results)
        return results
    
    def visualize_results(self, model_name, result_path):
        """Visualize results"""
        preds = self.trainer[model_name].predict(self.eval_dataset)
        y_pred = torch.argmax(torch.tensor(preds.predictions), axis=1)
        y_true = preds.label_ids
        label_names = self.train_dataset.features["label"].names
        
        # Print and save evaluation metrics
        print("Classification Report:")
        report = classification_report(y_true, y_pred)
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
                epochs = [log["epoch"] for log in history if "epoch" in log]
                train_acc = [log["train_accuracy"] for log in history if "train_accuracy" in log]
                eval_acc = [log["eval_accuracy"] for log in history if "eval_accuracy" in log]

                plt.figure(figsize=(10, 5))
                plt.plot(epochs, train_acc, label='Train Accuracy')
                plt.plot(epochs, eval_acc, label='Eval Accuracy')
                plt.xlabel('Epochs')
                plt.ylabel('Accuracy')
                plt.title('Training and Evaluation Accuracy Over Epochs')
                plt.legend()
                plt.grid(True)
                plt.savefig(f"{result_path}/accuracy_over_epochs.png")
                plt.close()
        
        return cm
