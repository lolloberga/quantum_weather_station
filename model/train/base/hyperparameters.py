
class Hyperparameters:

    def __init__(self, hyperparams: dict):
        self.hyperparameters = hyperparams
        super().__init__()

    def __getitem__(self, key):
        if key in self.hyperparameters:
            return self.hyperparameters[key]
        return None