import enum
import os
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Tuple
import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from model.train.base.hyperparameters import Hyperparameters
from model.train.hyperparams.default_hyperparams import DefaultHyperparameters


class Trainer(ABC):

    def __init__(self, model: nn.Module, name: str, criterion=None, optimizer: torch.optim.Optimizer = None,
                 writer: SummaryWriter = None, hyperparameters: Hyperparameters = None) -> None:
        super().__init__()
        self._writer = writer
        self._model = model
        self._criterion = criterion
        self._optim = optimizer
        self._name = name
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self._device)

        if writer is None:
            self._writer = SummaryWriter(os.path.join(os.getcwd(), 'runs', f"{self._name} - {datetime.today().strftime('%Y-%m-%d %H:%M')}"))
        if hyperparameters is None:
            self._hyperparameters = DefaultHyperparameters().hyperparameters
        else:
            self._hyperparameters = hyperparameters.hyperparameters

    @property
    def model(self):
        return self._model

    @property
    def writer(self) -> SummaryWriter:
        return self._writer

    '''
    @property
    def criterion(self):
        return self._criterion

    @criterion.setter
    def criterion(self, value) -> None:
        self._criterion = value

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        return self._optim

    @optimizer.setter
    def optimizer(self, value: torch.optim.Optimizer) -> None:
        self._optim = value

    @writer.setter
    def writer(self, value: SummaryWriter) -> None:
        self._writer = value
    '''

    @property
    def hyperparameters(self) -> enum.Enum:
        return self._hyperparameters

    @hyperparameters.setter
    def hyperparameters(self, value: Hyperparameters) -> None:
        self._hyperparameters = value

    @abstractmethod
    def get_optmizer(self) -> torch.optim.Optimizer:
        pass

    @abstractmethod
    def get_criterion(self):
        pass

    @abstractmethod
    def train(self, X_train: torch.Tensor, y_train: torch.Tensor, X_test: torch.Tensor = None,
              y_test: torch.Tensor = None) -> Tuple[np.array, np.array]:
        pass

    @abstractmethod
    def predict(self, X: torch.Tensor) -> np.array:
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass
