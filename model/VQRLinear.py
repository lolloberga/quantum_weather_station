import os

import pandas as pd
from pennylane import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch import nn
import pennylane as qml
from torch.utils.data import DataLoader

from config.config_parser import ConfigParser
from model.train.VQR_trainer import VQR_trainer
from model.train.hyperparams.vqr_hyperparams import VQR_Hyperparameters
from utils.dataset_utils import DatasetUtils

# Define constants to test the model
START_DATE_BOARD = '2022-11-03'
END_DATE_BOARD = '2023-06-15'
RANDOM_STATE = 42
N_QUBITS = 4  # one qubit for each input feature


class VQRLinearModel(nn.Module):

    def __init__(self, n_qubit: int, layers: int = 1, duplicate_qubits: bool = False) -> None:
        super().__init__()
        self.n_qubit = n_qubit * 2 if duplicate_qubits else n_qubit
        # initialize thetas (or weights) of NN
        shape = qml.StronglyEntanglingLayers.shape(n_layers=layers, n_wires=self.n_qubit)
        initial_weights = np.pi * np.random.random(shape, requires_grad=True)
        self.weights = nn.Parameter(torch.from_numpy(initial_weights), requires_grad=True)
        # initialize bias of NN
        self.bias = nn.Parameter(torch.from_numpy(np.zeros(1)), requires_grad=True)

    def encoder(self, x):
        qml.AngleEmbedding(x, wires=range(self.n_qubit))

    def layer(self, weights):
        qml.StronglyEntanglingLayers(weights, wires=range(self.n_qubit))

    def circuit(self, x):
        self.encoder(x)
        qml.Barrier(wires=range(self.n_qubit), only_visual=True)
        self.layer(self.weights)
        qml.Barrier(wires=range(self.n_qubit), only_visual=True)
        return qml.expval(qml.PauliZ(wires=0))

    def forward(self, X):
        # define the characteristics of the device
        dev = qml.device("default.qubit", wires=self.n_qubit)
        vqc = qml.QNode(self.circuit, dev, interface="torch")
        res = []
        for x in X:
            res.append(vqc(x) + self.bias)
        res = torch.stack(res)
        return res

    def draw_circuit(self, style: str = 'pennylane'):
        fig, _ = qml.draw_mpl(self.circuit, decimals=2, style=style, wire_order=range(self.n_qubit))(
            [x.item() for x in np.random.random(self.n_qubit, requires_grad=False)])
        return fig


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

    return train_loader, test_loader, df_arpa, scaler


def main() -> None:
    # Get project configurations
    cfg = ConfigParser()
    # Get default VQR's hyperparameters
    hyperparams = VQR_Hyperparameters().hyperparameters
    # Prepare dataset
    train_loader, test_loader, df_arpa, scaler = build_dataset(cfg, batch_loader_size=hyperparams['BATCH_SIZE'],
                                                               train_size=hyperparams['TRAIN_SIZE'])
    # Instantiate the model
    model = VQRLinearModel(N_QUBITS, layers=hyperparams['NUM_LAYERS'],
                              duplicate_qubits=hyperparams['DUPLICATE_QUBITS'])
    # Instantiate the trainer
    trainer = VQR_trainer(model, name='VQR_linear_model')
    train_losses, test_losses = trainer.train_loader(train_loader, test_loader)
    # Plot the train loss and test loss per iteration
    fig = trainer.draw_train_test_loss(train_losses, test_losses)
    trainer.save_image('VQR linear - Train and test loss', fig)
