import itertools
import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch import nn

from config.config_parser import ConfigParser
from model.LSTM import MyLSTM
from model.loss_functions.RMSELoss import RMSELoss
from model.train.LSTM_trainer import LSTM_trainer
from model.train.hyperparams.lstm_hyperparams import LSTM_Hyperparameters
from utils.dataset_utils import DatasetUtils

# Define constants
START_DATE_BOARD = '2022-11-03'
END_DATE_BOARD = '2023-06-15'
RANDOM_STATE = 42


def build_dataset(cfg: ConfigParser, hyperparams: dict) -> tuple:
    df_sensors = pd.read_csv(
        os.path.join(cfg.consts['DATASET_PATH'], 'unique_timeseries_by_median_hours.csv'))
    df_sensors.timestamp = pd.to_datetime(df_sensors.timestamp)
    df_sensors.timestamp += pd.Timedelta(hours=1)
    df_arpa = DatasetUtils.build_arpa_dataset(os.path.join(cfg.consts['DATASET_PATH'], 'arpa',
                                              'Dati PM10_PM2.5_2020-2022.csv')
                                 , os.path.join(cfg.consts['DATASET_PATH'], 'arpa',
                                                'Torino-Rubino_Polveri-sottili_2023-01-01_2023-06-30.csv'), START_DATE_BOARD, END_DATE_BOARD)

    df = df_sensors.merge(df_arpa, left_on=['timestamp'], right_on=['timestamp'])
    df.rename(columns={"data": "x", "pm25": "y"}, inplace=True)
    # Slide ARPA data 1 hour plus
    df['y'] = DatasetUtils.slide_plus_1hours(df['y'], df['x'][0])
    # Add month column and transform it into one-hot-encording
    # df['month'] = df.timestamp.dt.month
    # df['period_day'] = df['timestamp'].map(get_period_of_the_day)
    # Transform some features into one-hot encoding
    # df = pd.get_dummies(df, columns=['month', 'period_day'])

    input_data = df.drop(['timestamp'], axis=1)
    targets = df.y.values
    D = input_data.shape[1]  # Dimensionality of the input
    N = len(input_data) - hyperparams['T']

    train_size = int(len(input_data) * hyperparams['TRAIN_SIZE'])
    # Preparing X_train and y_train
    X_train = np.zeros((train_size, hyperparams['T'], D))
    y_train = np.zeros((train_size, 1))
    for t in range(train_size):
        X_train[t, :, :] = input_data[t:t + hyperparams['T']]
        y_train[t] = (targets[t + hyperparams['T']])

    # Preparing X_test and y_test
    X_test = np.zeros((N - train_size, hyperparams['T'], D))
    y_test = np.zeros((N - train_size, 1))
    for i in range(N - train_size):
        t = i + train_size
        X_test[i, :, :] = input_data[t:t + hyperparams['T']]
        y_test[i] = (targets[t + hyperparams['T']])

    # Make inputs and targets
    X_train = torch.from_numpy(X_train.astype(np.float32))
    y_train = torch.from_numpy(y_train.astype(np.float32))
    X_test = torch.from_numpy(X_test.astype(np.float32))
    y_test = torch.from_numpy(y_test.astype(np.float32))

    return X_train, X_test, y_train, y_test, D, df


def choose_optimizer(hyperparams: dict, model: nn.Module) -> torch.optim:
    optimizer = None
    if hyperparams['OPTIMIZER'] is not None and isinstance(hyperparams['OPTIMIZER'], str):
        if hyperparams['OPTIMIZER'].lower() == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams['LEARNING_RATE'])
        elif hyperparams['OPTIMIZER'].lower() == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=hyperparams['LEARNING_RATE'],
                                        momentum=hyperparams['MOMENTUM'], weight_decay=hyperparams['WEIGHT_DECAY'])
        else:
            raise ValueError(f'Unknown optimizer {hyperparams["OPTIMIZER"]}')
    return optimizer


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


def runs(hyperparams: LSTM_Hyperparameters,
         name: str = 'LSTM_TUNING_test') -> None:
    # Get project configurations
    cfg = ConfigParser()
    # Prepare dataset
    X_train, X_test, y_train, y_test, D, df = build_dataset(cfg, hyperparams.hyperparameters)
    # Instantiate the model
    model = MyLSTM(D, hyperparams['HIDDEN_SIZE'], 2, hyperparams['OUTPUT_SIZE'])
    # Get the correct optimizer and criterion
    optimizer = choose_optimizer(hyperparams.hyperparameters, model)
    criterion = choose_criterion(hyperparams.hyperparameters)
    # Instantiate the trainer
    trainer = LSTM_trainer(model, name=name, hyperparameters=hyperparams, optimizer=optimizer, criterion=criterion)
    train_losses, test_losses = trainer.train(X_train, y_train, X_test, y_test)
    # Save hparams result on Tensorboard
    trainer.writer.add_hparams(hyperparams.hyperparameters,
                               {'loss/train': train_losses[-1], 'loss/test': test_losses[-1]})
    # Plot the train loss and test loss per iteration
    fig = trainer.draw_train_test_loss(train_losses, test_losses)
    trainer.save_image(f'{name} - Train and test loss', fig)


def main():
    print('Start LSTM hyperparameters tuning')
    # Get configuration space
    epochs = [200, 300, 450]
    lr = [0.01, 0.0001, 0.001]
    h1 = [512, 200, 600]
    t = [5, 1, 3, 10]
    optimizer = ['adam', 'sgd']
    criterion = ['rmse', 'mse', 'l1']
    hparam_names = ['NUM_EPOCHS', 'LEARNING_RATE', 'HIDDEN_SIZE', 'T',
                    'OPTIMIZER', 'CRITERION']
    # Get all possibile hyperaprameters combination
    combinations = list(itertools.product(epochs, lr, h1, t, optimizer, criterion))
    hparams = []
    for _, comb in enumerate(combinations):
        hparam = dict()
        for idx, param in enumerate(comb):
            hparam[hparam_names[idx]] = param
        hparams.append(LSTM_Hyperparameters(hparam))
    # Iterate over all possible combinations of hyperparameters
    print(f'Fine-tuning of {len(hparams)} combinations')
    for hparam in hparams:
        runs(hparam, name=f'LSTM_TUNING_{datetime.today().strftime("%Y%m%d_%H%M%S")}')
    print('End of LSTM hyperparameters tuning')


if __name__ == "__main__":
    main()
