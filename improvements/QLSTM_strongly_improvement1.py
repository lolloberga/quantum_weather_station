import itertools
import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt

from config.config_parser import ConfigParser
from model.QLSTM import QLSTM
from model.train.QLSTM_trainer import QLSTM_trainer
from model.train.hyperparams.qlstm_hyperparams import QLSTM_Hyperparameters
from utils.dataset_utils import DatasetUtils
from utils.tuning_utils import TuningUtils

"""
    THIS PYTHON SCRIPT IS RELATED TO TEST A PARTICULAR COMBINATION OF DATASET AND HYPERPARAMS.
    THE GOALS IS TO TAKE THE BEST QLSTM MODEL AND TRAIN IT WITH MORE THAN 300 EPOCHS.
"""

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
                                                             'Torino-Rubino_Polveri-sottili_2023-01-01_2023-06-30.csv'),
                                              START_DATE_BOARD, END_DATE_BOARD)

    df = df_sensors.merge(df_arpa, left_on=['timestamp'], right_on=['timestamp'])
    df.rename(columns={"data": "x", "pm25": "y"}, inplace=True)
    # Slide ARPA data 1 hour plus
    df['y'] = DatasetUtils.slide_plus_1hours(df['y'], df['x'][0])

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


def runs(hyperparams: QLSTM_Hyperparameters,
         name: str = 'QLSTM_STRONGLY_TUNING_test') -> None:
    # Get project configurations
    cfg = ConfigParser()
    # Prepare dataset
    X_train, X_test, y_train, y_test, D, df = build_dataset(cfg, hyperparams.hyperparameters)
    # Instantiate the model
    model = QLSTM(D, hidden_size=hyperparams['HIDDEN_SIZE'], n_qubits=hyperparams['N_QUBITS'],
                  n_qlayers=hyperparams['N_QLAYERS'], batch_first=True, ansatz='strongly')
    # Get the correct optimizer and criterion
    optimizer = TuningUtils.choose_optimizer(hyperparams.hyperparameters, model)
    criterion = TuningUtils.choose_criterion(hyperparams.hyperparameters)
    # Instantiate the trainer
    trainer = QLSTM_trainer(model, name=name, hyperparameters=hyperparams, optimizer=optimizer, criterion=criterion)
    train_losses, test_losses = trainer.train(X_train, y_train, X_test, y_test)
    # Save hparams result on Tensorboard
    trainer.writer.add_hparams(hyperparams.hyperparameters,
                               {'loss/train': train_losses.min(), 'loss/test': test_losses.min()})
    # Plot the train loss and test loss per iteration
    fig = trainer.draw_train_test_loss(train_losses, test_losses)
    trainer.save_image(f'{name} - Train and test loss', fig)
    # Plot the model performance
    test_target = y_test.cpu().detach().numpy()
    test_predictions = []
    for i in range(len(test_target)):
        input_ = X_test[i].reshape(1, hyperparams['T'], D)
        p = model(input_)[0, 0].item()
        test_predictions.append(p)

    plot_len = len(test_predictions)
    plot_df = df[['timestamp', 'y']].copy(deep=True)
    plot_df = plot_df.iloc[-plot_len:]
    plot_df['pred'] = test_predictions
    plot_df.set_index('timestamp', inplace=True)

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(plot_df['y'], label='ARPA pm25', linewidth=1)
    ax.plot(plot_df['pred'], label='Predicted pm25', linewidth=1)
    ax.set_xlabel('timestamp')
    ax.set_ylabel(r'$\mu g/m^3$')
    ax.set_title(f'QLSTM Performance - {hyperparams["NUM_EPOCHS"]} epochs - T = {hyperparams["T"]}')
    ax.legend(loc='lower right')
    fig.tight_layout()
    trainer.save_image(f'{name} - Performance', fig)


def main() -> None:
    print('Start QLSTM strongly entangler improvement #1')
    # Get configuration space
    epochs = [300]
    lr = [0.01, 0.001]
    h1 = [15]
    q_layers = [5, 8]
    qubits = [7, 10]
    t = [5]
    optimizer = ['adam', 'rmsprop']
    criterion = ['l1', 'rmse']
    hparam_names = ['NUM_EPOCHS', 'LEARNING_RATE', 'HIDDEN_SIZE', 'N_QLAYERS', 'N_QUBITS', 'T',
                    'OPTIMIZER', 'CRITERION']
    # Get all possible hyperparameters combination
    combinations = list(itertools.product(epochs, lr, h1, q_layers, qubits, t, optimizer, criterion))
    hparams = []
    for _, comb in enumerate(combinations):
        hparam = dict()
        for idx, param in enumerate(comb):
            hparam[hparam_names[idx]] = param
        hparams.append(QLSTM_Hyperparameters(hparam))
    # Iterate over all possible combinations of hyperparameters
    print(f'Fine-tuning of {len(hparams)} combinations')
    for hparam in hparams:
        runs(hparam, name=f'QLSTM_STRONGLY_POWERED_{datetime.today().strftime("%Y%m%d_%H%M%S")}')
    print('End of QLSTM strongly improvement #1')
