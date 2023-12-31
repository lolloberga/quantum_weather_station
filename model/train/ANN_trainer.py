import time
from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model.train.base.hyperparameters import Hyperparameters
from model.train.base.trainer import Trainer
from model.train.hyperparams.ann_hyperparams import ANN_Hyperparameters
from utils.tensorboard_utils import TensorboardUtils


class ANN_trainer(Trainer):

    def __init__(self, model: nn.Module, name: str = None, criterion=None, optimizer: torch.optim.Optimizer = None,
                 writer: SummaryWriter = None, hyperparameters: Hyperparameters = None) -> None:
        self._name = name
        if hyperparameters is None:
            hyperparameters = ANN_Hyperparameters()

        super().__init__(model, self.get_name(), criterion, optimizer, writer, hyperparameters)

    def get_optimizer(self) -> torch.optim.Optimizer:
        if self._optim is None:
            self._optim = torch.optim.Adam(self.model.parameters(), lr=self.hyperparameters['LEARNING_RATE'])
        return self._optim

    def get_criterion(self):
        if self._criterion is None:
            self._criterion = nn.MSELoss()
        return self._criterion

    def train_loader(self, train_loader: DataLoader, test_loader: DataLoader) -> Tuple[np.array, np.array]:

        train_losses = np.zeros(self.hyperparameters['NUM_EPOCHS'])
        test_losses = np.zeros(self.hyperparameters['NUM_EPOCHS'])
        start = time.time()

        for epoch in tqdm(range(self.hyperparameters['NUM_EPOCHS']), desc=f'Training the {self.get_name()} model'):
            self.model.train()
            current_loss = 0.0

            optimizer = self.get_optimizer()
            criterion = self.get_criterion()

            for i, data in enumerate(train_loader, 0):
                inputs, targets = data
                inputs, targets = inputs.float().to(self.device), targets.float().to(self.device)
                targets = targets.reshape((targets.shape[0], 1))
                # Zero the gradients
                optimizer.zero_grad()
                # Forward pass
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                # Backpropagation
                loss.backward()
                optimizer.step()
                current_loss += loss.item()
                # Write the network graph at epoch 0, batch 0
                if epoch == 0 and i == 0:
                    self.writer.add_graph(self.model, input_to_model=data[0], verbose=False)

            train_losses[epoch] = current_loss / len(train_loader)
            self.writer.add_scalar(self.get_name() + " - Loss/train", current_loss / len(train_loader), epoch)

            # Evaluate accuracy at end of each epoch
            self.model.eval()
            val_loss = 0.0
            val_steps = 0
            for i, data in enumerate(test_loader, 0):
                inputs, targets = data
                inputs, targets = inputs.float().to(self.device), targets.float().to(self.device)
                targets = targets.reshape((targets.shape[0], 1))
                y_pred = self.model(inputs)
                test_loss = criterion(y_pred, targets)
                val_loss += test_loss.item()
                val_steps += 1

            test_losses[epoch] = val_loss / len(test_loader)
            self.writer.add_scalar(self.get_name() + " - Loss/test", val_loss / len(test_loader), epoch)

            # Draw plot predicted vs actual (tensorboard)
            if (epoch + 1) % 30 == 0:
                y_pred = torch.from_numpy(self.predict(test_loader.dataset.X)).reshape(-1, 1)
                self.writer.add_figure(self.get_name() + ' - Predicted vs Actual',
                                       TensorboardUtils.draw_prediction_tensorboard(y_pred, test_loader.dataset.y, epoch),
                                       global_step=epoch+1)

        # Save the model at the end of the training (for future inference)
        self._save_model()
        self.writer.flush()
        self.writer.close()
        # Send notification if needed
        self._send_notification('END_OF_TRAINING', {'#TIME#': round(time.time()-start, 2)})
        return train_losses, test_losses

    def train(self, X_train: torch.Tensor, y_train: torch.Tensor, X_test: torch.Tensor = None,
              y_test: torch.Tensor = None) -> Tuple[np.array, np.array]:
        pass

    def predict(self, X: torch.Tensor) -> np.array:
        self.model.eval()
        test_predictions = []
        with torch.no_grad():
            for i in range(len(X)):
                input_ = X[i].float()
                y_pred = self.model(input_)
                test_predictions.append(y_pred.item())
        return np.array(test_predictions)

    def get_name(self) -> str:
        return 'ANN' if self._name is None else self._name
