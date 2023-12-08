
from model.train.base.hyperparameters import Hyperparameters


LSTM_defaults_hyperparams = {
    'TRAIN_SIZE': 0.7,
    'LEARNING_RATE': 0.01,
    'NUM_EPOCHS': 300,
    'HIDDEN_SIZE': 512,
    'OUTPUT_SIZE': 1,
    'MOMENTUM': 0.9,
    'WEIGHT_DECAY': 1e-4,
    'T': 5  # Number of hours to look while predicting
}


class LSTM_Hyperparameters(Hyperparameters):

    def __init__(self, hyperparams: dict = None, append_to_other: bool = True):
        if hyperparams is None:
            hyperparams = LSTM_defaults_hyperparams
        else:
            if append_to_other:
                default_hyperparams = LSTM_defaults_hyperparams
                for k, v in hyperparams.items():
                    default_hyperparams[k] = v
                hyperparams = default_hyperparams.copy()
        super().__init__(hyperparams)
