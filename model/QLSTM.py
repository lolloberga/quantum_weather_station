import os
import random

import numpy as np
import pandas as pd
import pennylane as qml
import torch
import torch.nn as nn
from matplotlib import pyplot as plt

from config.config_parser import ConfigParser
from model.train.QLSTM_trainer import QLSTM_trainer
from model.train.hyperparams.qlstm_hyperparams import QLSTM_Hyperparameters
from utils.dataset_utils import DatasetUtils

# Define constants
START_DATE_BOARD = '2022-11-03'
END_DATE_BOARD = '2023-06-15'
np.random.seed = 7
torch.manual_seed(7)
random.seed(7)


class QLSTM(nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_size: int = 3,
                 n_qubits: int = 4,
                 n_qlayers: int = 2,
                 batch_first: bool = True,
                 return_sequences: bool = False,
                 return_state: bool = False,
                 backend: str = "default.qubit.torch",
                 ansatz: str = "basic"):
        super(QLSTM, self).__init__()
        self.n_inputs = input_size
        self.hidden_size = hidden_size
        self.concat_size = self.n_inputs + self.hidden_size
        self.n_qubits = n_qubits
        self.n_qlayers = n_qlayers
        self.backend = backend  # "default.qubit", "qiskit.basicaer", "qiskit.ibm"
        self.fc1 = nn.Linear(self.hidden_size, 1)

        self.batch_first = batch_first
        self.return_sequences = return_sequences
        self.return_state = return_state

        # self.dev = qml.device("default.qubit", wires=self.n_qubits)
        # self.dev = qml.device('qiskit.basicaer', wires=self.n_qubits)
        # self.dev = qml.device('qiskit.ibm', wires=self.n_qubits)
        # use 'qiskit.ibmq' instead to run on hardware

        self.wires_forget = [f"wire_forget_{i}" for i in range(self.n_qubits)]
        self.wires_input = [f"wire_input_{i}" for i in range(self.n_qubits)]
        self.wires_update = [f"wire_update_{i}" for i in range(self.n_qubits)]
        self.wires_output = [f"wire_output_{i}" for i in range(self.n_qubits)]

        self.dev_forget = qml.device(self.backend, wires=self.wires_forget, torch_device='cpu')
        self.dev_input = qml.device(self.backend, wires=self.wires_input, torch_device='cpu')
        self.dev_update = qml.device(self.backend, wires=self.wires_update, torch_device='cpu')
        self.dev_output = qml.device(self.backend, wires=self.wires_output, torch_device='cpu')

        print(f'Choose "{ansatz}" type of ansatz')

        def _circuit_forget(inputs, weights):
            qml.templates.AngleEmbedding(inputs, wires=self.wires_forget)
            qml.templates.BasicEntanglerLayers(weights, wires=self.wires_forget)
            return [qml.expval(qml.PauliZ(wires=w)) for w in self.wires_forget]

        def _circuit_forget_strongly(inputs, weights):
            qml.templates.AngleEmbedding(inputs, wires=self.wires_forget)
            qml.templates.StronglyEntanglingLayers(weights, wires=self.wires_forget)
            return [qml.expval(qml.PauliZ(wires=w)) for w in self.wires_forget]

        self.qlayer_forget = qml.QNode(_circuit_forget_strongly if ansatz == 'strongly' else _circuit_forget,
                                       self.dev_forget, interface="torch", diff_method="backprop")

        def _circuit_input(inputs, weights):
            qml.templates.AngleEmbedding(inputs, wires=self.wires_input)
            qml.templates.BasicEntanglerLayers(weights, wires=self.wires_input)
            return [qml.expval(qml.PauliZ(wires=w)) for w in self.wires_input]

        def _circuit_input_strongly(inputs, weights):
            qml.templates.AngleEmbedding(inputs, wires=self.wires_input)
            qml.templates.StronglyEntanglingLayers(weights, wires=self.wires_input)
            return [qml.expval(qml.PauliZ(wires=w)) for w in self.wires_input]

        self.qlayer_input = qml.QNode(_circuit_input_strongly if ansatz == 'strongly' else _circuit_input,
                                      self.dev_input, interface="torch", diff_method="backprop")

        def _circuit_update(inputs, weights):
            qml.templates.AngleEmbedding(inputs, wires=self.wires_update)
            qml.templates.BasicEntanglerLayers(weights, wires=self.wires_update)
            return [qml.expval(qml.PauliZ(wires=w)) for w in self.wires_update]

        def _circuit_update_strongly(inputs, weights):
            qml.templates.AngleEmbedding(inputs, wires=self.wires_update)
            qml.templates.StronglyEntanglingLayers(weights, wires=self.wires_update)
            return [qml.expval(qml.PauliZ(wires=w)) for w in self.wires_update]

        self.qlayer_update = qml.QNode(_circuit_update_strongly if ansatz == 'strongly' else _circuit_update,
                                       self.dev_update, interface="torch", diff_method="backprop")

        def _circuit_output(inputs, weights):
            qml.templates.AngleEmbedding(inputs, wires=self.wires_output)
            qml.templates.BasicEntanglerLayers(weights, wires=self.wires_output)
            return [qml.expval(qml.PauliZ(wires=w)) for w in self.wires_output]

        def _circuit_output_strongly(inputs, weights):
            qml.templates.AngleEmbedding(inputs, wires=self.wires_output)
            qml.templates.StronglyEntanglingLayers(weights, wires=self.wires_output)
            return [qml.expval(qml.PauliZ(wires=w)) for w in self.wires_output]

        self.qlayer_output = qml.QNode(_circuit_output_strongly if ansatz == 'strongly' else _circuit_output,
                                       self.dev_output, interface="torch", diff_method="backprop")

        if ansatz == 'strongly':
            shape = qml.StronglyEntanglingLayers.shape(n_layers=self.n_qlayers, n_wires=self.n_qubits)
            weight_shapes = {"weights": shape}
        else:
            weight_shapes = {"weights": (n_qlayers, n_qubits)}

        # print(f"weight_shapes = (n_qlayers, n_qubits) = ({n_qlayers}, {n_qubits})")

        self.clayer_in = torch.nn.Linear(self.concat_size, n_qubits)
        '''
        self.VQC = {
            'forget': qml.qnn.TorchLayer(self.qlayer_forget, weight_shapes),
            'input': qml.qnn.TorchLayer(self.qlayer_input, weight_shapes),
            'update': qml.qnn.TorchLayer(self.qlayer_update, weight_shapes),
            'output': qml.qnn.TorchLayer(self.qlayer_output, weight_shapes)
        }
        '''
        self.vqc_forget = qml.qnn.TorchLayer(self.qlayer_forget, weight_shapes)
        self.vqc_input = qml.qnn.TorchLayer(self.qlayer_input, weight_shapes)
        self.vqc_update = qml.qnn.TorchLayer(self.qlayer_update, weight_shapes)
        self.vqc_output = qml.qnn.TorchLayer(self.qlayer_output, weight_shapes)
        self.clayer_out = torch.nn.Linear(self.n_qubits, self.hidden_size)
        # self.clayer_out = [torch.nn.Linear(n_qubits, self.hidden_size) for _ in range(4)]

    def forward(self, x, init_states=None):
        """
        x.shape is (batch_size, seq_length, feature_size)
        recurrent_activation -> sigmoid
        activation -> tanh
        """
        if self.batch_first is True:
            batch_size, seq_length, features_size = x.size()
        else:
            seq_length, batch_size, features_size = x.size()

        hidden_seq = []
        if init_states is None:
            h_t = torch.zeros(batch_size, self.hidden_size)  # hidden state (output)
            c_t = torch.zeros(batch_size, self.hidden_size)  # cell state
        else:
            # for now we ignore the fact that in PyTorch you can stack multiple RNNs
            # so we take only the first elements of the init_states tuple init_states[0][0], init_states[1][0]
            h_t, c_t = init_states
            h_t = h_t[0]
            c_t = c_t[0]

        for t in range(seq_length):
            # get features from the t-th element in seq, for all entries in the batch
            x_t = x[:, t, :]

            # Concatenate input and hidden state
            v_t = torch.cat((h_t, x_t), dim=1)

            # match qubit dimension
            y_t = self.clayer_in(v_t)

            f_t = torch.sigmoid(self.clayer_out(self.vqc_forget(y_t)))  # forget block
            i_t = torch.sigmoid(self.clayer_out(self.vqc_input(y_t)))  # input block
            g_t = torch.tanh(self.clayer_out(self.vqc_update(y_t)))  # update block
            o_t = torch.sigmoid(self.clayer_out(self.vqc_output(y_t)))  # output block

            c_t = (f_t * c_t) + (i_t * g_t)
            h_t = o_t * torch.tanh(c_t)

            hidden_seq.append(h_t.unsqueeze(0))
        hidden_seq = torch.cat(hidden_seq, dim=0)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        # return hidden_seq, (h_t, c_t)
        x = self.fc1(h_t)
        return x


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


def main() -> None:
    # Get project configurations
    cfg = ConfigParser()
    hyperparams = QLSTM_Hyperparameters()
    # Prepare dataset
    X_train, X_test, y_train, y_test, D, df = build_dataset(cfg, hyperparams.hyperparameters)
    # Instantiate the model
    model = QLSTM(D, hidden_size=hyperparams['HIDDEN_SIZE'], n_qubits=hyperparams['N_QUBITS'],
                  n_qlayers=hyperparams['N_QLAYERS'], batch_first=True, backend='default.qubit.torch')
    # Instantiate the trainer
    trainer = QLSTM_trainer(model, name='QLSTM_test', hyperparameters=hyperparams)
    train_losses, test_losses = trainer.train(X_train, y_train, X_test, y_test)
    # Plot the train loss and test loss per iteration
    fig = trainer.draw_train_test_loss(train_losses, test_losses)
    trainer.save_image('QLSTM_test - Train and test loss', fig)

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
    trainer.save_image('QLSTM_test - Performance', fig)
