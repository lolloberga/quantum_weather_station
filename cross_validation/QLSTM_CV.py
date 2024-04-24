import os

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold

from config.config_parser import ConfigParser
from model.QLSTM import QLSTM
from model.train.QLSTM_trainer import QLSTM_trainer
from model.train.hyperparams.qlstm_hyperparams import QLSTM_Hyperparameters
from utils.dataset_utils import DatasetUtils
from utils.tuning_utils import TuningUtils

START_DATE_BOARD = '2022-11-03'
END_DATE_BOARD = '2023-06-15'
RANDOM_STATE = 42
NUM_SPLIT = 5


def build_dataset(cfg: ConfigParser) -> tuple:
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

    return input_data, targets


def prepare_train_split(input_data: np.ndarray, targets: np.ndarray, hyperparams: dict) -> tuple:
    D = input_data.shape[1]  # Dimensionality of the input
    train_size = len(input_data) - hyperparams['T']
    # Preparing X_train and y_train
    X_train = np.zeros((train_size, hyperparams['T'], D))
    y_train = np.zeros((train_size, 1))
    for t in range(train_size):
        X_train[t, :, :] = input_data[t:t + hyperparams['T']]
        y_train[t] = (targets[t + hyperparams['T']])
    # Make inputs and targets
    X_train = torch.from_numpy(X_train.astype(np.float32))
    y_train = torch.from_numpy(y_train.astype(np.float32))
    return X_train, y_train


def prepare_test_split(input_data: np.ndarray, targets: np.ndarray, hyperparams: dict) -> tuple:
    D = input_data.shape[1]  # Dimensionality of the input
    test_size = len(input_data) - hyperparams['T']
    # Preparing X_test and y_test
    X_test = np.zeros((test_size, hyperparams['T'], D))
    y_test = np.zeros((test_size, 1))
    for t in range(test_size):
        X_test[t, :, :] = input_data[t:t + hyperparams['T']]
        y_test[t] = (targets[t + hyperparams['T']])
    # Make inputs and targets
    X_test = torch.from_numpy(X_test.astype(np.float32))
    y_test = torch.from_numpy(y_test.astype(np.float32))
    return X_test, y_test


def main() -> None:
    print('Start QLSTM cross-validation')
    # Get project configurations
    cfg = ConfigParser()
    # Set the best hyperparams combination
    hyperparams = QLSTM_Hyperparameters(
        {
            'TRAIN_SIZE': 0.15,
            'LEARNING_RATE': 0.01,
            'OPTIMIZER': 'adam',
            'CRITERION': 'l1',
            'HIDDEN_SIZE': 15,
            'NUM_EPOCHS': 400,
            'T': 5,
            'N_QUBITS': 7,
            'N_QLAYERS': 5
        }
    )
    # Prepare dataset
    input_data, targets = build_dataset(cfg)
    # Prepare cross-validation
    kf = KFold(n_splits=NUM_SPLIT, shuffle=False)
    X = input_data.values
    # Loop fo each fold
    for i, (train_index, test_index) in enumerate(kf.split(X, targets)):
        name = f"QLSTM_CV_FOLD{i + 1}"
        # Prepare the final dataset
        X_train, y_train = prepare_train_split(input_data.iloc[train_index], targets[train_index],
                                               hyperparams.hyperparameters)
        X_test, y_test = prepare_test_split(input_data.iloc[test_index], targets[test_index],
                                            hyperparams.hyperparameters)
        # Instantiate the model
        model = QLSTM(input_data.shape[1], hidden_size=hyperparams['HIDDEN_SIZE'], n_qubits=hyperparams['N_QUBITS'],
                      n_qlayers=hyperparams['N_QLAYERS'], batch_first=True, ansatz='strongly')
        # Get the optimizer and criterion
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

    print('End of QLSTM cross-validation')
