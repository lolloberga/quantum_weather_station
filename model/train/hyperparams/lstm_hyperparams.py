import enum

from model.train.base.hyperparameters import Hyperparameters


class _Defaults(enum.Enum):
    LEARNING_RATE = 0.01
    NUM_EPOCHS = 300
    HIDDEN_SIZE = 512
    OUTPUT_SIZE = 1
    T = 5  # Number of hours to look while predicting


class LSTM_Hyperparameters(Hyperparameters):

    def __init__(self):
        super().__init__(_Defaults)
