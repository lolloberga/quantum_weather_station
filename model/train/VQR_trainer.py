import time
from typing import Tuple

import numpy as np
import torch
from torch import nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model.loss_functions.RMSELoss import RMSELoss
from model.train.base.hyperparameters import Hyperparameters
from model.train.base.trainer import Trainer
from model.train.hyperparams.vqr_hyperparams import VQR_Hyperparameters


class VQR_trainer(Trainer):

    def __init__(self, model: nn.Module, name: str, criterion=None, optimizer: torch.optim.Optimizer = None,
                 writer: SummaryWriter = None, hyperparameters: Hyperparameters = None) -> None:

        self._name = name
        if hyperparameters is None:
            hyperparameters = VQR_Hyperparameters()
        super().__init__(model, self.get_name(), criterion, optimizer, writer, hyperparameters)
        # Draw the circuit
        self.writer.add_figure(self.get_name() + ' - Circuit', model.draw_circuit(), global_step=0)

    def get_optimizer(self) -> torch.optim.Optimizer:
        if self._optim is None:
            self._optim = torch.optim.Adam(self.model.parameters(), lr=self.hyperparameters['LEARNING_RATE'])
        return self._optim

    def get_criterion(self):
        if self._criterion is None:
            self._criterion = RMSELoss()
        return self._criterion

    def train(self, X_train: torch.Tensor, y_train: torch.Tensor, X_test: torch.Tensor = None,
              y_test: torch.Tensor = None) -> Tuple[np.array, np.array]:
        pass

    def train_loader(self, train_loader: DataLoader, test_loader: DataLoader) -> Tuple[np.array, np.array]:
        train_losses = np.zeros(self.hyperparameters['NUM_EPOCHS'])
        test_losses = np.zeros(self.hyperparameters['NUM_EPOCHS'])
        start = time.time()

        for epoch in tqdm(range(self.hyperparameters['NUM_EPOCHS']), desc=f'Training the {self.get_name()} model'):
            current_loss = 0.0

            optimizer = self.get_optimizer()
            criterion = self.get_criterion()

            for i, (X, y) in enumerate(train_loader):
                self.model.train()
                # Zero the gradients
                optimizer.zero_grad()
                # Forward pass
                y_pred = self.model(X).squeeze()
                # calculate the loss
                loss = criterion(y_pred, y)
                # update the parameters
                loss.backward()
                optimizer.step()
                current_loss += loss.item()

            train_losses[epoch] = current_loss / len(train_loader)
            self.writer.add_scalar(self.get_name() + " - Loss/train", current_loss / len(train_loader), epoch)

            # Evaluate accuracy at end of each epoch
            self.model.eval()
            current_loss = 0.0
            with torch.inference_mode():
                for X_t, y_t in test_loader:
                    # Zero the gradients
                    optimizer.zero_grad()
                    # Forward pass
                    y_pred = self.model(X_t).squeeze()
                    # calculate the loss
                    loss = criterion(y_pred, y_t)
                    current_loss += loss.item()

                test_losses[epoch] = current_loss / len(test_loader)
                self.writer.add_scalar(self.get_name() + " - Loss/test", current_loss / len(test_loader), epoch)

        # Save the model at the end of the training (for future inference)
        self._save_model()
        self.writer.flush()
        self.writer.close()
        # Send notification if needed
        self._send_notification('END_OF_TRAINING', {'#TIME#': round(time.time() - start, 2)})
        return train_losses, test_losses

    def predict(self, X: torch.Tensor) -> np.array:
        self.model.eval()
        test_predictions = []
        with torch.no_grad():
            for i in range(len(X)):
                input_ = X[i].float()
                y_pred = self.model(input_).squeeze()
                test_predictions.append(y_pred.item())
        return np.array(test_predictions)

    def get_name(self) -> str:
        return 'VQR' if self._name is None else self._name
