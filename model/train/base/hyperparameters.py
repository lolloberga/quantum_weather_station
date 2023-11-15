import enum


class Hyperparameters:

    def __init__(self, hyperparams: enum.Enum):
        self.hyperparameters = hyperparams
        super().__init__()

    def get_value(self, obj):
        if isinstance(obj, Hyperparameters):
            return obj.hyperparameters.value
