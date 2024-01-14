from model.train.base.hyperparameters import Hyperparameters

VQR_defaults_hyperparams = {
    'TRAIN_SIZE': 0.75,
    'LEARNING_RATE': 0.01,
    'NUM_EPOCHS': 1,
    'BATCH_SIZE': 10,
    'NUM_LAYERS': 4,
    'DUPLICATE_QUBITS': False,
    'OPTIMIZER': 'adam',
    'CRITERION': 'rmse',
    'MOMENTUM': 0.9,
    'WEIGHT_DECAY': 0
}


class VQR_Hyperparameters(Hyperparameters):

    def __init__(self, hyperparams: dict = None, append_to_other: bool = True):
        if hyperparams is None:
            hyperparams = VQR_defaults_hyperparams
        else:
            if append_to_other:
                default_hyperparams = VQR_defaults_hyperparams
                for k, v in hyperparams.items():
                    default_hyperparams[k] = v
                hyperparams = default_hyperparams.copy()
        super().__init__(hyperparams)
