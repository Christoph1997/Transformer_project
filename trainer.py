from transformers import TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, f1_score
import torch

class Trainer_instance:
    def __init__(self, model, tokenized_dataset):
        self.model = model
        self.train_dataset = tokenized_dataset["train"]
        self.eval_dataset = tokenized_dataset["test"]
        self.trainer = None
        self.training_args = None

    def set_training_arguments(self, epochs=2, batch_size=16):
        """Training arguments"""
        self.training_args = TrainingArguments(
            output_dir="./bert-amazon-polarity",
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_steps=100,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            save_total_limit=1,
            remove_unused_columns=False,
            report_to="none",  # disables W&B or TensorBoard
        )

    def train(self):
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
        self.trainer = trainer

    @staticmethod
    def compute_metrics(p):
        """Define evaluation metrics"""
        preds = torch.argmax(torch.tensor(p.predictions), axis=1)
        labels = p.label_ids
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds)
        return {"accuracy": acc, "f1": f1}

    def evaluate(self):
        """Evaluate"""
        results = self.trainer.evaluate()
        print("Evaluation results:", results)
