from transformers import AutoModelForSequenceClassification

# Load model
class Model:
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = None

    def load_model(self, dataset_name):
        # Load model
        if dataset_name == "McAuley-Lab/Amazon-Reviews-2023":
            # Load model with 5 labels for Amazon Reviews dataset
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=5)
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=2)