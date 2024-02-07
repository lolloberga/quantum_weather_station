import torch
from torch import nn

from model.loss_functions.RMSELoss import RMSELoss


class TuningUtils:

    @staticmethod
    def choose_criterion(hyperparams: dict):
        criterion = None
        if hyperparams['CRITERION'] is not None and isinstance(hyperparams['CRITERION'], str):
            if hyperparams['CRITERION'].lower() == 'mse':
                criterion = nn.MSELoss()
            elif hyperparams['CRITERION'].lower() == 'l1':
                criterion = nn.L1Loss()
            elif hyperparams['CRITERION'].lower() == 'rmse':
                criterion = RMSELoss()
            else:
                raise ValueError(f'Unknown criterion {hyperparams["CRITERION"]}')
        return criterion

    @staticmethod
    def choose_optimizer(hyperparams: dict, model: nn.Module) -> torch.optim:
        optimizer = None
        if hyperparams['OPTIMIZER'] is not None and isinstance(hyperparams['OPTIMIZER'], str):
            if hyperparams['OPTIMIZER'].lower() == 'adam':
                optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams['LEARNING_RATE'])
            elif hyperparams['OPTIMIZER'].lower() == 'sgd':
                optimizer = torch.optim.SGD(model.parameters(), lr=hyperparams['LEARNING_RATE'],
                                            momentum=hyperparams['MOMENTUM'], weight_decay=hyperparams['WEIGHT_DECAY'])
            elif hyperparams['OPTIMIZER'].lower() == 'rmsprop':
                optimizer = torch.optim.RMSprop(model.parameters(), lr=hyperparams['LEARNING_RATE'],
                                            momentum=hyperparams['MOMENTUM'], weight_decay=hyperparams['WEIGHT_DECAY'])
            else:
                raise ValueError(f'Unknown optimizer {hyperparams["OPTIMIZER"]}')
        return optimizer
