import enum
import os
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from config import ConfigParser
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
        print(f"{self._name} - Using {self._device} device")
        self._model.to(self._device)
        # Get project configurations
        self._cfg = ConfigParser()

        if writer is None:
            self._writer = SummaryWriter(
                os.path.join(os.getcwd(), 'runs', f"{self._name} - {datetime.today().strftime('%Y-%m-%d %H:%M')}"))
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

    @property
    def device(self):
        return self._device

    @property
    def hyperparameters(self) -> enum.Enum:
        return self._hyperparameters

    @hyperparameters.setter
    def hyperparameters(self, value: Hyperparameters) -> None:
        self._hyperparameters = value

    def _save_model(self) -> None:
        folder = self._cfg.consts['MODEL_CHECKPOINT_PATH']
        if not os.path.exists(folder):
            os.makedirs(folder)
        folder = os.path.join(self._cfg.consts['MODEL_CHECKPOINT_PATH'], self.get_name())
        if not os.path.exists(folder):
            os.makedirs(folder)

        torch.save(self._model.state_dict(), os.path.join(folder, f"{self.get_name()}_{datetime.today().strftime('%Y-%m-%d_%H-%M')}.pt"))

    def save_image(self, name: str, fig: plt.Figure) -> None:
        folder = self._cfg.consts['MODEL_DRAWS_PATH']
        if not os.path.exists(folder):
            os.makedirs(folder)
        folder = os.path.join(self._cfg.consts['MODEL_DRAWS_PATH'], self.get_name())
        if not os.path.exists(folder):
            os.makedirs(folder)

        fig.savefig(os.path.join(folder, f"{name} - {datetime.today().strftime('%Y-%m-%d_%H-%M')}.png"))

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
    def train_loader(self, train_loader: DataLoader, test_loader: DataLoader, use_ray_tune: bool = False) \
            -> Tuple[np.array, np.array]:
        pass

    @abstractmethod
    def predict(self, X: torch.Tensor) -> np.array:
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass

    def draw_train_test_loss(self, train_losses: np.array, test_losses: np.array):
        # Plot the train loss and test loss per iteration
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        ax.plot(train_losses, label='Train loss')
        ax.plot(test_losses, label='Test loss')
        ax.set_xlabel('epoch no')
        ax.set_ylabel('loss')
        ax.set_title(
            f'Train/Test loss at each iteration - {self.hyperparameters.NUM_EPOCHS.value} epochs')
        ax.legend()
        fig.tight_layout()
        return fig
