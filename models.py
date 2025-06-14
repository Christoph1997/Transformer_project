from transformers import AutoModelForSequenceClassification

# Load model
class Model:
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = None

    def load_model(self):
        # Load model
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=2)