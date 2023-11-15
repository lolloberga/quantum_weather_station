import os
from datetime import datetime
from typing import Tuple

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model.train.base.hyperparameters import Hyperparameters
from model.train.base.trainer import Trainer
from model.train.hyperparams.lstm_hyperparams import LSTM_Hyperparameters


def draw_prediction_tensorboard(prediction: torch.Tensor, actual: torch.Tensor, epoch: int) -> plt.Figure:
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(actual.detach().numpy().T[0], label='ARPA pm25', linewidth=1)
    ax.plot(prediction.detach().numpy().T[0], label='Predicted pm25', linewidth=1)
    ax.set_xlabel('')
    ax.set_ylabel(r'$\mu g/m^3$')
    ax.set_title(f'LSTM Performance - At epoch {epoch}')
    ax.legend(loc='lower right')
    fig.tight_layout()
    return fig


class LSTM_trainer(Trainer):

    def __init__(self, model: nn.Module, criterion=None, optimizer: torch.optim.Optimizer = None,
                 writer: SummaryWriter = None, hyperparameters: Hyperparameters = None) -> None:

        if hyperparameters is None:
            hyperparameters = LSTM_Hyperparameters()

        super().__init__(model, self.get_name(), criterion, optimizer, writer, hyperparameters)

    def get_name(self) -> str:
        return 'lstm_approach_1'

    def get_optmizer(self) -> torch.optim.Optimizer:
        if self._optim is None:
            self._optim = torch.optim.SGD(self.model.parameters(), lr=self.hyperparameters.LEARNING_RATE.value,
                                          momentum=0.9, weight_decay=1e-4)
        return self._optim

    def get_criterion(self):
        if self._criterion is None:
            self._criterion = nn.MSELoss()
        return self._criterion

    def _save_model(self) -> None:
        torch.save(self.model.state_dict(), os.path.join(os.getcwd(), 'model', 'checkpoints',
                                                         f"lstm_approach_1_{datetime.today().strftime('%Y-%m-%d_%H-%M')}.pt"))

    def train(self, X_train: torch.Tensor, y_train: torch.Tensor, X_test: torch.Tensor = None,
              y_test: torch.Tensor = None) -> Tuple[np.array, np.array]:

        # Draw the graph
        self.writer.add_graph(self.model, X_train)

        self.model.train()
        train_losses = np.zeros(self.hyperparameters.NUM_EPOCHS.value)
        test_losses = np.zeros(self.hyperparameters.NUM_EPOCHS.value)

        X_train, y_train = X_train.to(self._device), y_train.to(self._device)
        X_test, y_test = X_test.to(self._device), y_test.to(self._device)

        optimizer = self.get_optmizer()
        criterion = self.get_criterion()

        for epoch in tqdm(range(self.hyperparameters.NUM_EPOCHS.value), desc='Train the LSTM model'):
            optimizer.zero_grad()

            # Forward pass
            outputs = self.model(X_train)
            loss = criterion(outputs, y_train)
            self.writer.add_scalar("Loss/train", loss, epoch)

            # Backpropagation
            loss.backward()
            optimizer.step()

            # Train loss
            train_losses[epoch] = loss.item()

            # Test loss
            test_outputs = self.model(X_test)
            test_loss = criterion(test_outputs, y_test)
            self.writer.add_scalar("Loss/test", test_loss, epoch)
            test_losses[epoch] = test_loss.item()

            # Draw plot predicted vs actuals (tensorboard)
            if (epoch + 1) % 5 == 0:
                self.writer.add_figure('LSTM - Predicted vs Actual',
                                       draw_prediction_tensorboard(test_outputs, y_test, epoch), global_step=epoch)

        # Save the model at the end of the training (for future inference)
        self._save_model()
        self.writer.flush()
        self.writer.close()
        return train_losses, test_losses

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
            f'Train loss at each iteration - {self.hyperparameters.NUM_EPOCHS.value} epochs - T = {self.hyperparameters.T.value}')
        ax.legend()
        fig.tight_layout()
        return fig
