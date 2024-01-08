import itertools
import os
from datetime import datetime
from pennylane import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader

from config.config_parser import ConfigParser
from model.VQRLinear import VQRLinearModel
from model.train.VQR_trainer import VQR_trainer
from model.train.hyperparams.vqr_hyperparams import VQR_Hyperparameters
from utils.dataset_utils import DatasetUtils
from utils.tuning_utils import TuningUtils

# Define constants to test the model
START_DATE_BOARD = '2022-11-03'
END_DATE_BOARD = '2023-06-15'
RANDOM_STATE = 42
N_QUBITS = 4  # one qubit for each input feature


def build_dataset(cfg: ConfigParser, batch_loader_size: int = 10, train_size: float = 0.7) -> tuple:
    df_sensors = pd.read_csv(
        os.path.join(cfg.consts['DATASET_PATH'], 'unique_timeseries_by_median_hours_all_attributes.csv'))
    df_sensors.timestamp = pd.to_datetime(df_sensors.timestamp)
    df_sensors.timestamp += pd.Timedelta(hours=1)
    df_arpa = DatasetUtils.build_arpa_dataset(os.path.join(cfg.consts['DATASET_PATH'], 'arpa',
                                                           'Dati PM10_PM2.5_2020-2022.csv')
                                              , os.path.join(cfg.consts['DATASET_PATH'], 'arpa',
                                                             'Torino-Rubino_Polveri-sottili_2023-01-01_2023-06-30.csv'),
                                              START_DATE_BOARD, END_DATE_BOARD)
    df = df_sensors.merge(df_arpa, left_on=['timestamp'], right_on=['timestamp'])
    df.rename(columns={"pm25_x": "x", "pm25_y": "y"}, inplace=True)
    # Slide ARPA data 1 hour plus
    df['y'] = DatasetUtils.slide_plus_1hours(df['y'], df['x'][0])
    df.drop(['timestamp'], axis=1, inplace=True)
    # Scale the features between -1 and +1
    scaler = MinMaxScaler(feature_range=(-1, 1))
    rescaledX = scaler.fit_transform(df)
    df = pd.DataFrame(rescaledX, columns=df.columns)
    # Split dataset
    X = df.loc[:, df.columns != "y"]
    y = df['y']
    X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, train_size=train_size,
                                                        shuffle=False,
                                                        random_state=RANDOM_STATE)
    X_train = torch.from_numpy(X_train).type(torch.double)
    X_test = torch.from_numpy(X_test).type(torch.double)
    y_train = torch.from_numpy(np.array(y_train, requires_grad=False)).type(torch.double)
    y_test = torch.from_numpy(np.array(y_test, requires_grad=False)).type(torch.double)

    train_tensor = torch.utils.data.TensorDataset(X_train, y_train)
    test_tensor = torch.utils.data.TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_tensor, batch_size=batch_loader_size, shuffle=True)
    test_loader = DataLoader(test_tensor, batch_size=batch_loader_size, shuffle=False)

    return train_loader, test_loader


def runs(hyperparams: VQR_Hyperparameters,
         name: str = 'VQR_LINEAR_test') -> None:
    # Get project configurations
    cfg = ConfigParser()
    # Prepare dataset
    train_loader, test_loader = build_dataset(cfg, batch_loader_size=hyperparams['BATCH_SIZE'],
                                              train_size=hyperparams['TRAIN_SIZE'])
    # Instantiate the model
    model = VQRLinearModel(N_QUBITS, layers=hyperparams['NUM_LAYERS'],
                           duplicate_qubits=hyperparams['DUPLICATE_QUBITS'])
    # Get the correct optimizer and criterion
    optimizer = TuningUtils.choose_optimizer(hyperparams.hyperparameters, model)
    criterion = TuningUtils.choose_criterion(hyperparams.hyperparameters)
    # Instantiate the trainer
    trainer = VQR_trainer(model, name=name, hyperparameters=hyperparams, optimizer=optimizer, criterion=criterion)
    train_losses, test_losses = trainer.train_loader(train_loader, test_loader)
    # Save hparams result on Tensorboard
    trainer.writer.add_hparams(hyperparams.hyperparameters,
                               {'loss/train': train_losses.min(), 'loss/test': test_losses.min()})
    # Plot the train loss and test loss per iteration
    fig = trainer.draw_train_test_loss(train_losses, test_losses)
    trainer.save_image(f'{name} - Train and test loss', fig)


def main():
    print('Start VQR linear hyperparameters tuning')
    # Get configuration space
    epochs = [200, 400, 500]
    batches = [4, 10, 15]
    lr = [0.0001, 0.001, 0.01]
    layers = [3, 4, 5]
    optimizer = ['adam', 'sgd']
    criterion = ['rmse', 'mse', 'l1']
    hparam_names = ['NUM_EPOCHS', 'BATCH_SIZE', 'LEARNING_RATE', 'LAYERS', 'OPTIMIZER', 'CRITERION']
    # Get all possible hyperaprameters combination
    combinations = list(itertools.product(epochs, batches, lr, layers, optimizer, criterion))
    hparams = []
    for _, comb in enumerate(combinations):
        hparam = dict()
        for idx, param in enumerate(comb):
            hparam[hparam_names[idx]] = param
        hparams.append(VQR_Hyperparameters(hparam))
    # Iterate over all possible combinations of hyperparameters
    print(f'Fine-tuning of {len(hparams)} combinations')
    for hparam in hparams:
        runs(hparam, name=f'VQR_LINEAR_TUNING_{datetime.today().strftime("%Y%m%d_%H%M%S")}')
    print('End of VQR linear hyperparameters tuning')
