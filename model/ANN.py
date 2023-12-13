import os.path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datetime import datetime
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config.config_parser import ConfigParser
from model.train.ANN_trainer import ANN_trainer
from model.train.hyperparams.ann_hyperparams import ANN_Hyperparameters
from utils.dataset_utils import DatasetUtils

# Define constants
START_DATE_BOARD = '2022-11-03'
END_DATE_BOARD = '2023-06-15'


class PM25AnnDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, batch_size):
        self._batch_size = batch_size
        self.X = X
        self.y = y
        if not torch.is_tensor(X) and not torch.is_tensor(y):
            self.X = torch.from_numpy(X)
            self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i * self._batch_size: (i + 1) * self._batch_size], self.y[i]


class PM25AnnDataset2(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        if not torch.is_tensor(X) and not torch.is_tensor(y):
            self.X = torch.from_numpy(X)
            self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


class MyNeuralNetwork(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_size: int, hidden_size_2: int = 90,
                 hidden_size_3: int = 30):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size_2),
            nn.ReLU(),
            nn.Linear(hidden_size_2, hidden_size_3),
            nn.ReLU(),
            nn.Linear(hidden_size_3, output_size),
        )

    def forward(self, x):
        out = self.net(x)
        return out


def build_dataset(cfg: ConfigParser, hyperparams: dict) -> tuple:
    df_sensors = pd.read_csv(
        os.path.join(os.getcwd(), 'resources', 'dataset', 'unique_timeseries_by_median_minutes.csv'))
    df_sensors.timestamp = pd.to_datetime(df_sensors.timestamp)
    df_sensors.timestamp += pd.Timedelta(minutes=1)
    df_arpa = DatasetUtils.build_arpa_dataset(os.path.join(os.getcwd(), 'resources', 'dataset', 'arpa',
                                                           'Dati PM10_PM2.5_2020-2022.csv')
                                              , os.path.join(os.getcwd(), 'resources', 'dataset', 'arpa',
                                                             'Torino-Rubino_Polveri-sottili_2023-01-01_2023-06-30.csv'),
                                              START_DATE_BOARD, END_DATE_BOARD)
    # Apply date range filter (inner join)
    mask = (df_arpa['timestamp'] >= min(df_sensors.timestamp) + pd.DateOffset(hours=1)) & (
            df_arpa['timestamp'] <= max(df_sensors.timestamp))
    df_arpa = df_arpa.loc[mask]
    mask = (df_sensors['timestamp'] >= min(df_arpa.timestamp) - pd.DateOffset(hours=1)) & (
            df_sensors['timestamp'] <= max(df_arpa.timestamp))
    df_sensors = df_sensors.loc[mask]
    # Slide ARPA data 1 hour plus
    df_arpa.reset_index(inplace=True)
    df_arpa['pm25'] = DatasetUtils.slide_plus_1hours(df_arpa['pm25'], df_arpa['pm25'][0])
    # Unique dataset
    df = df_sensors.merge(df_arpa, how='left', on='timestamp').drop(['index'], axis=1)

    X = df['data'].values
    y = df['pm25'].dropna().values
    X_train = X[: int(len(X) * hyperparams['TRAIN_SIZE'])]
    X_test = X[int(len(X) * hyperparams['TRAIN_SIZE']):]
    y_train = y[: int(len(y) * hyperparams['TRAIN_SIZE'])]
    y_test = y[int(len(y) * hyperparams['TRAIN_SIZE']):]
    # Convert to 2D PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)
    # Convert into my dataset cutom class
    train_dataset = PM25AnnDataset(X_train, y_train, hyperparams['INPUT_SIZE'])
    test_dataset = PM25AnnDataset(X_test, y_test, hyperparams['INPUT_SIZE'])
    # Use data-loader in order to have batches
    train_loader = DataLoader(train_dataset, batch_size=hyperparams['BATCH_SIZE'], shuffle=False, num_workers=0,
                              sampler=SequentialSampler(train_dataset))
    test_loader = DataLoader(test_dataset, batch_size=hyperparams['BATCH_SIZE'], shuffle=False, num_workers=0,
                             sampler=SequentialSampler(test_dataset))
    return train_loader, test_loader, df_arpa


def build_dataset_2(cfg: ConfigParser, hyperparams: dict) -> tuple:
    df_sensors = pd.read_csv(
        os.path.join(os.getcwd(), 'resources', 'dataset', 'unique_timeseries_by_median_minutes.csv'))
    df_sensors.timestamp = pd.to_datetime(df_sensors.timestamp)
    df_sensors.timestamp += pd.Timedelta(minutes=1)
    df_arpa = DatasetUtils.build_arpa_dataset(os.path.join(os.getcwd(), 'resources', 'dataset', 'arpa',
                                                           'Dati PM10_PM2.5_2020-2022.csv')
                                              , os.path.join(os.getcwd(), 'resources', 'dataset', 'arpa',
                                                             'Torino-Rubino_Polveri-sottili_2023-01-01_2023-06-30.csv'),
                                              START_DATE_BOARD, END_DATE_BOARD)
    # Apply date range filter (inner join)
    mask = (df_arpa['timestamp'] >= min(df_sensors.timestamp) + pd.DateOffset(hours=1)) & (
            df_arpa['timestamp'] <= max(df_sensors.timestamp))
    df_arpa = df_arpa.loc[mask]
    mask = (df_sensors['timestamp'] >= min(df_arpa.timestamp) - pd.DateOffset(hours=1)) & (
            df_sensors['timestamp'] <= max(df_arpa.timestamp))
    df_sensors = df_sensors.loc[mask]
    # Slide ARPA data 1 hour plus
    df_arpa.reset_index(inplace=True)
    df_arpa['pm25'] = DatasetUtils.slide_plus_1hours(df_arpa['pm25'], df_arpa['pm25'][0])
    # Unique dataset
    columns = ["tm{}".format(i) for i in range(1, 61)]
    columns.insert(0, 'arpa')
    df = pd.DataFrame(columns=columns)

    df_sensors.reset_index(inplace=True, drop=True)
    for i, arpa in enumerate(df_arpa['pm25']):
        row = df_sensors['data'][i * 60: (i + 1) * 60].values
        row = np.append(arpa, row)
        df.loc[len(df)] = row.tolist()

    X = df.loc[:, df.columns != "arpa"]
    y = df['arpa']
    X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, train_size=hyperparams['TRAIN_SIZE'],
                                                        shuffle=False,
                                                        random_state=hyperparams['RANDOM_STATE'])
    # Convert to 2D PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)
    # Convert into my dataset cutom class
    train_dataset = PM25AnnDataset2(X_train, y_train)
    test_dataset = PM25AnnDataset2(X_test, y_test)
    # Use data-loader in order to have batches
    train_loader = DataLoader(train_dataset, batch_size=hyperparams['BATCH_SIZE'], shuffle=False, num_workers=0,
                              sampler=RandomSampler(train_dataset))
    test_loader = DataLoader(test_dataset, batch_size=hyperparams['BATCH_SIZE'], shuffle=False, num_workers=0,
                             sampler=RandomSampler(test_dataset))
    return train_loader, test_loader, df_arpa


def main():
    # Get project configurations
    cfg = ConfigParser()
    hyperparams = ANN_Hyperparameters().hyperparameters
    # Prepare dataset
    train_loader, test_loader, df_arpa = build_dataset_2(cfg, hyperparams)
    # Instantiate the model
    model = MyNeuralNetwork(hyperparams['INPUT_SIZE'], hyperparams['OUTPUT_SIZE'], hyperparams['HIDDEN_SIZE'])
    # Instantiate the trainer
    trainer = ANN_trainer(model, name='ANN')
    train_losses, test_losses = trainer.train_loader(train_loader, test_loader)
    # Plot the train loss and test loss per iteration
    fig = trainer.draw_train_test_loss(train_losses, test_losses)
    trainer.save_image('ANN - Train and test loss', fig)

    # Plot the model performance
    test_target = test_loader.dataset.y.cpu().detach().numpy()
    test_predictions = []
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(len(test_target)), desc='Preparing predictions'):
            input_ = test_loader.dataset.X[i].float()
            y_pred = model(input_)
            test_predictions.append(y_pred.item())

    plot_len = len(test_predictions)
    plot_df = df_arpa[['timestamp', 'pm25']].copy(deep=True)
    plot_df = plot_df.iloc[-plot_len:]
    plot_df['pred'] = test_predictions
    plot_df.set_index('timestamp', inplace=True)
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(plot_df['pm25'], label='ARPA pm25', linewidth=1)
    ax.plot(plot_df['pred'], label='Predicted pm25', linewidth=1)
    ax.set_xlabel('timestamp')
    ax.set_ylabel(r'$\mu g/m^3$')
    ax.set_title(f'ANN Performance - {hyperparams["NUM_EPOCHS"]} epochs')
    ax.legend(loc='lower right')
    fig.tight_layout()
    trainer.save_image('ANN - Performance', fig)
