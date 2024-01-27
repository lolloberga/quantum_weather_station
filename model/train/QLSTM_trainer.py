import time
from typing import Tuple

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model.train.base.hyperparameters import Hyperparameters
from model.train.base.trainer import Trainer
from model.train.hyperparams.qlstm_hyperparams import QLSTM_Hyperparameters
from utils.tensorboard_utils import TensorboardUtils


class QLSTM_trainer(Trainer):

    def __init__(self, model: nn.Module, name: str = None, criterion=None, optimizer: torch.optim.Optimizer = None,
                 writer: SummaryWriter = None, hyperparameters: Hyperparameters = None) -> None:

        self._name = name
        if hyperparameters is None:
            hyperparameters = QLSTM_Hyperparameters()

        super().__init__(model, self.get_name(), criterion, optimizer, writer, hyperparameters)

    def get_name(self) -> str:
        return 'QLSTM_model' if self._name is None else self._name

    def get_optimizer(self) -> torch.optim.Optimizer:
        if self._optim is None:
            self._optim = torch.optim.SGD(self.model.parameters(), lr=self.hyperparameters['LEARNING_RATE'],
                                          momentum=self.hyperparameters['MOMENTUM'], weight_decay=1e-4)
        return self._optim

    def get_criterion(self):
        if self._criterion is None:
            self._criterion = nn.MSELoss()
        return self._criterion

    def train(self, X_train: torch.Tensor, y_train: torch.Tensor, X_test: torch.Tensor = None,
              y_test: torch.Tensor = None) -> Tuple[np.array, np.array]:

        train_losses = np.zeros(self.hyperparameters['NUM_EPOCHS'])
        test_losses = np.zeros(self.hyperparameters['NUM_EPOCHS'])
        start = time.time()

        X_train, y_train = X_train.to(self.device), y_train.to(self.device)
        X_test, y_test = X_test.to(self.device), y_test.to(self.device)

        optimizer = self.get_optimizer()
        criterion = self.get_criterion()

        for epoch in tqdm(range(self.hyperparameters['NUM_EPOCHS']), desc=f'Training the {self.get_name()} model'):
            self.model.train()
            optimizer.zero_grad()

            # Forward pass
            outputs = self.model(X_train)
            loss = criterion(outputs, y_train)
            self.writer.add_scalar(self.get_name() + " - Loss/train", loss, epoch)

            # Backpropagation
            loss.backward()
            optimizer.step()

            # Train loss
            train_losses[epoch] = loss.item()

            # Test loss
            self.model.eval()
            with torch.inference_mode():
                test_outputs = self.model(X_test)
                test_loss = criterion(test_outputs, y_test)
            self.writer.add_scalar(self.get_name() + " - Loss/test", test_loss, epoch)
            test_losses[epoch] = test_loss.item()

            # Draw plot predicted vs actual (tensorboard)
            """if (epoch + 1) % 30 == 0:
                self.writer.add_figure(self.get_name() + ' - Predicted vs Actual',
                                       TensorboardUtils.draw_prediction_tensorboard(test_outputs, y_test, epoch),
                                       global_step=epoch+1)"""

        # Save the model at the end of the training (for future inference)
        self._save_model()
        self.writer.flush()
        self.writer.close()
        # Send notification if needed
        self._send_notification('END_OF_TRAINING', {'#TIME#': round(time.time()-start, 2)})
        return train_losses, test_losses

    def train_loader(self, train_loader: DataLoader, test_loader: DataLoader, use_ray_tune: bool = False) \
            -> Tuple[np.array, np.array]:
        pass

    def predict(self, X: torch.Tensor) -> np.array:
        self.model.eval()
        # Generate predictions for the test dataset
        predictions = []
        with torch.no_grad():
            # Forward pass
            outputs = self.model(X)
            # Save the predictions
            predictions += outputs.squeeze().tolist()
        predictions = np.array(predictions)
        return predictions

    def draw_train_test_loss(self, train_losses: np.array, test_losses: np.array):
        # Plot the train loss and test loss per iteration
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        ax.plot(train_losses, label='Train loss')
        ax.plot(test_losses, label='Test loss')
        ax.set_xlabel('epoch no')
        ax.set_ylabel('loss')
        ax.set_title(
            f'Train/Test loss at each iteration - {self.hyperparameters["NUM_EPOCHS"]} epochs - T = {self.hyperparameters["T"]}')
        ax.legend()
        fig.tight_layout()
        return fig