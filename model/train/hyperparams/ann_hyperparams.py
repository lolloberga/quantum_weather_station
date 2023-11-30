import enum

from model.train.base.hyperparameters import Hyperparameters


class _Defaults(enum.Enum):
    TRAIN_SIZE = 0.7
    RANDOM_STATE = 42
    INPUT_SIZE = 60
    HIDDEN_SIZE = 128
    HIDDEN_SIZE_2 = 90
    HIDDEN_SIZE_3 = 30
    OUTPUT_SIZE = 1
    LEARNING_RATE = 0.0001
    NUM_EPOCHS = 20
    BATCH_SIZE = 2


class ANN_Hyperparameters(Hyperparameters):

    def __init__(self, hyperparams: enum.Enum = None, dict_params: dict = None):
        if dict_params is not None:
            default_dict = Hyperparameters.from_enum_to_dict(_Defaults)
            for k, v in dict_params.items():
                default_dict[k] = v
            hyperparams = Hyperparameters.from_dict_to_enum('ANN_Hyperparameters', default_dict)
        if hyperparams is None:
            hyperparams = _Defaults()
        super().__init__(hyperparams)



