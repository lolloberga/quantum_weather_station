import enum

from model.train.base.hyperparameters import Hyperparameters


class _Defaults(enum.Enum):
    TRAIN_SIZE = 0.7
    RANDOM_STATE = 42
    INPUT_SIZE = 60
    HIDDEN_SIZE = 128
    OUTPUT_SIZE = 1
    LEARNING_RATE = 0.0001
    NUM_EPOCHS = 20
    BATCH_SIZE = 2


class ANN_Hyperparameters(Hyperparameters):

    def __init__(self, hyperparams: enum.Enum = None):
        if hyperparams is None:
            hyperparams = _Defaults
        super().__init__(hyperparams)



