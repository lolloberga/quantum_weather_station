
from model.train.base.hyperparameters import Hyperparameters


QLSTM_defaults_hyperparams = {
    'TRAIN_SIZE': 0.7,
    'LEARNING_RATE': 0.01,
    'NUM_EPOCHS': 10,
    'HIDDEN_SIZE': 3,
    'OUTPUT_SIZE': 1,
    'MOMENTUM': 0.9,
    'WEIGHT_DECAY': 1e-4,
    'T': 5,  # Number of hours to look while predicting.
    'N_QUBITS': 4,
    'N_QLAYERS': 2
}


class QLSTM_Hyperparameters(Hyperparameters):

    def __init__(self, hyperparams: dict = None, append_to_other: bool = True):
        if hyperparams is None:
            hyperparams = QLSTM_defaults_hyperparams
        else:
            if append_to_other:
                default_hyperparams = QLSTM_defaults_hyperparams
                for k, v in hyperparams.items():
                    default_hyperparams[k] = v
                hyperparams = default_hyperparams.copy()
        super().__init__(hyperparams)
