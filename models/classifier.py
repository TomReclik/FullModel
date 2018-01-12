import keras
import InceptionV3

class classifier:
    def __init__(self, weights_path, model, SEM, centroids):
        self.models = ["InceptionV3", "EERACN"]
        if not model in self.models:
            raise "Specified model "
