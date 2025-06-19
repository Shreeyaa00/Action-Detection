import random

class ActionClassifier:
    def __init__(self):
        self.labels = ["standing", "walking", "fighting", "falling", "running", "loitering", "robbery"]

    def classify(self, person_crop):
        # Placeholder: Replace with real model inference
        return random.choice(self.labels)
