import itertools
import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm

from config.config_parser import ConfigParser
from model.ANN import PM25AnnDataset2, MyNeuralNetwork
from model.train.ANN_trainer import ANN_trainer
from model.train.base.trainer import Trainer
from model.train.hyperparams.ann_hyperparams import ANN_Hyperparameters
from utils.dataset_utils import DatasetUtils
from utils.tuning_utils import TuningUtils

# Define constants
START_DATE_BOARD = '2022-11-03'
END_DATE_BOARD = '2023-06-15'
RANDOM_STATE = 42


def build_dataset_2(cfg: ConfigParser, batch_loader_size: int = 2, train_size: float = 0.7) -> tuple:
    df_sensors = pd.read_csv(
        os.path.join(cfg.consts['DATASET_PATH'], 'unique_timeseries_by_median_minutes.csv'))
    df_sensors.timestamp = pd.to_datetime(df_sensors.timestamp)
    df_sensors.timestamp += pd.Timedelta(minutes=1)
    df_arpa = DatasetUtils.build_arpa_dataset(os.path.join(cfg.consts['DATASET_PATH'], 'arpa',
                                                           'Dati PM10_PM2.5_2020-2022.csv')
                                              , os.path.join(cfg.consts['DATASET_PATH'], 'arpa',
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
    X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, train_size=train_size,
                                                        shuffle=False,
                                                        random_state=RANDOM_STATE)
    # Convert to 2D PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)
    # Convert into my dataset cutom class
    train_dataset = PM25AnnDataset2(X_train, y_train)
    test_dataset = PM25AnnDataset2(X_test, y_test)

    # Use data-loader in order to have batches
    train_loader = DataLoader(train_dataset, batch_size=batch_loader_size, shuffle=False, num_workers=0,
                              sampler=RandomSampler(train_dataset))
    test_loader = DataLoader(test_dataset, batch_size=batch_loader_size, shuffle=False, num_workers=0,
                             sampler=RandomSampler(test_dataset))
    return train_loader, test_loader, df_arpa


def plot_performance(model: nn.Module, test_loader: DataLoader, df_arpa: pd.DataFrame, trainer: Trainer,
                     name: str) -> None:
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
    ax.set_title(name)
    ax.legend(loc='upper right')
    fig.tight_layout()
    trainer.save_image('ANN - Performance', fig)


def runs(hyperparams: ANN_Hyperparameters,
         name: str = 'ANN_TUNING_test') -> None:
    # Get project configurations
    cfg = ConfigParser()
    # Prepare dataset
    train_loader, test_loader, df_arpa = build_dataset_2(cfg, batch_loader_size=hyperparams['BATCH_SIZE'],
                                                         train_size=hyperparams['TRAIN_SIZE'])
    # Instantiate the model
    model = MyNeuralNetwork(60, 1, hyperparams['HIDDEN_SIZE'], hyperparams['HIDDEN_SIZE_2'],
                            hyperparams['HIDDEN_SIZE_3'])
    # Get the correct optimizer and criterion
    optimizer = TuningUtils.choose_optimizer(hyperparams.hyperparameters, model)
    criterion = TuningUtils.choose_criterion(hyperparams.hyperparameters)
    # Instantiate the trainer
    trainer = ANN_trainer(model, name=name, hyperparameters=hyperparams, optimizer=optimizer, criterion=criterion)
    train_losses, test_losses = trainer.train_loader(train_loader, test_loader)
    # Save hparams result on Tensorboard
    trainer.writer.add_hparams(hyperparams.hyperparameters,
                               {'loss/train': train_losses[-1], 'loss/test': test_losses[-1]})
    # Plot the train loss and test loss per iteration
    fig = trainer.draw_train_test_loss(train_losses, test_losses)
    trainer.save_image(f'{name} - Train and test loss', fig)


def main():
    print('Start ANN hyperparameters tuning')
    # Get configuration space
    epochs = [200, 400, 500]
    batches = [2, 4, 10, 15]
    lr = [0.0001, 0.001, 0.01]
    h1 = [128, 90, 60]
    h2 = [90, 40, 30]
    h3 = [30, 20, 10]
    optimizer = ['adam', 'sgd']
    criterion = ['rmse', 'mse', 'l1']
    hparam_names = ['NUM_EPOCHS', 'BATCH_SIZE', 'LEARNING_RATE', 'HIDDEN_SIZE', 'HIDDEN_SIZE_2', 'HIDDEN_SIZE_3',
                    'OPTIMIZER', 'CRITERION']
    # Get all possibile hyperaprameters combination
    combinations = list(itertools.product(epochs, batches, lr, h1, h2, h3, optimizer, criterion))
    hparams = []
    for _, comb in enumerate(combinations):
        hparam = dict()
        for idx, param in enumerate(comb):
            hparam[hparam_names[idx]] = param
        hparams.append(ANN_Hyperparameters(hparam))
    # Iterate over all possible combinations of hyperparameters
    print(f'Fine-tuning of {len(hparams)} combinations')
    for hparam in hparams:
        runs(hparam, name=f'ANN_TUNING_{datetime.today().strftime("%Y%m%d_%H%M%S")}')
    print('End of ANN hyperparameters tuning')

