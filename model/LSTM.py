from config import ConfigParser
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

from config import ConfigParser
from model.train.LSTM_trainer import LSTM_trainer
from model.train.hyperparams.lstm_hyperparams import LSTM_Hyperparameters

# Define constants
torch.manual_seed(1)
TRAIN_SIZE = 0.8
START_DATE_BOARD = '2022-11-03'
END_DATE_BOARD = '2023-06-15'
T = 5  # Number of hours to look while predicting


class MyLSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(MyLSTM, self).__init__()
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.M = hidden_dim
        self.L = layer_dim

        self.rnn = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=layer_dim,
            batch_first=True)
        # batch_first to have (batch_dim, seq_dim, feature_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, X):
        # initial hidden state and cell state
        h0 = torch.zeros(self.L, X.size(0), self.M).to(self._device)
        c0 = torch.zeros(self.L, X.size(0), self.M).to(self._device)

        out, (hn, cn) = self.rnn(X, (h0.detach(), c0.detach()))

        # h(T) at the final time step
        out = self.fc(out[:, -1, :])
        return out


def change_hour_format(hour: str) -> str:
    return hour + ":00" if len(hour.split(':')) <= 2 else hour


def build_arpa_dataset(arpa_2022: str, arpa_2023: str) -> pd.DataFrame:
    df_arpa_2022 = pd.read_csv(arpa_2022, sep=';')
    df_arpa_2023 = pd.read_csv(arpa_2023, sep=';', index_col=False)
    df_arpa_2022.dropna(inplace=True)
    df_arpa_2023 = df_arpa_2023[df_arpa_2023.Stato == 'V']

    df_arpa = pd.DataFrame(columns=['timestamp', 'pm25'])
    data_series_2022 = df_arpa_2022['Data'] + " " + df_arpa_2022['Ora'].map(lambda x: change_hour_format(x))
    data_series_2023 = df_arpa_2023['Data rilevamento'] + ' ' + df_arpa_2023['Ora'].map(lambda x: change_hour_format(x))
    pm25_series = df_arpa_2022['PM2.5']

    data_series = pd.concat([data_series_2022, data_series_2023], ignore_index=True)
    pm25_series = pd.concat([pm25_series, df_arpa_2023['Valore']], ignore_index=True)

    df_arpa['timestamp'] = data_series
    df_arpa['pm25'] = pm25_series
    df_arpa.timestamp = pd.to_datetime(df_arpa.timestamp, format="%d/%m/%Y %H:%M:%S")
    # Apply date range filter
    mask = (df_arpa['timestamp'] >= START_DATE_BOARD) & (df_arpa['timestamp'] <= END_DATE_BOARD)
    df_arpa = df_arpa.loc[mask]

    # Apply a special filter in which I remove all ARPA's values below 4
    df_arpa = df_arpa[df_arpa['pm25'] > 4]
    return df_arpa


def get_period_of_the_day(timestamp):
    h = timestamp.hour
    if 0 <= h <= 6:
        return 1
    elif 7 <= h <= 12:
        return 2
    elif 13 <= h <= 18:
        return 3
    else:
        return 4


def build_dataset(cfg: ConfigParser) -> tuple:
    df_sensors = pd.read_csv(
        os.path.join(os.getcwd(), 'resources', 'dataset', 'unique_timeserie_by_median.csv'))
    df_sensors.timestamp = pd.to_datetime(df_sensors.timestamp)
    df_arpa = build_arpa_dataset(os.path.join(os.getcwd(), 'resources', 'dataset', 'arpa',
                                              'Dati PM10_PM2.5_2020-2022.csv')
                                 , os.path.join(os.getcwd(), 'resources', 'dataset', 'arpa',
                                                'Torino-Rubino_Polveri-sottili_2023-01-01_2023-06-30.csv'))

    df = df_sensors.merge(df_arpa, left_on=['timestamp'], right_on=['timestamp'])
    df.rename(columns={"data": "x", "pm25": "y"}, inplace=True)
    # Add month column and transform it into one-hot-encording
    df['month'] = df.timestamp.dt.month
    df['period_day'] = df['timestamp'].map(get_period_of_the_day)
    # Transform some features into one-hot encoding
    df = pd.get_dummies(df, columns=['month', 'period_day'])

    input_data = df.drop(['timestamp', 'y'], axis=1)
    targets = df.y.values
    D = input_data.shape[1]  # Dimensionality of the input
    N = len(input_data) - T

    train_size = int(len(input_data) * TRAIN_SIZE)
    # Preparing X_train and y_train
    X_train = np.zeros((train_size, T, D))
    y_train = np.zeros((train_size, 1))
    for t in range(train_size):
        X_train[t, :, :] = input_data[t:t + T]
        y_train[t] = (targets[t + T])

    # Preparing X_test and y_test
    X_test = np.zeros((N - train_size, T, D))
    y_test = np.zeros((N - train_size, 1))
    for i in range(N - train_size):
        t = i + train_size
        X_test[i, :, :] = input_data[t:t + T]
        y_test[i] = (targets[t + T])

    # Make inputs and targets
    X_train = torch.from_numpy(X_train.astype(np.float32))
    y_train = torch.from_numpy(y_train.astype(np.float32))
    X_test = torch.from_numpy(X_test.astype(np.float32))
    y_test = torch.from_numpy(y_test.astype(np.float32))

    return X_train, X_test, y_train, y_test, D, df


def main():
    # Get project configurations
    cfg = ConfigParser()
    # Prepare dataset
    X_train, X_test, y_train, y_test, D, df = build_dataset(cfg)

    # Instantiate the model
    hyperparams = LSTM_Hyperparameters().hyperparameters
    model = MyLSTM(D, hyperparams.HIDDEN_SIZE.value, 2, hyperparams.OUTPUT_SIZE.value)
    # Instantiate the trainer
    trainer = LSTM_trainer(model)
    train_losses, test_losses = trainer.train(X_train, y_train, X_test, y_test)

    # Plot the train loss and test loss per iteration
    fig = trainer.draw_train_test_loss(train_losses, test_losses)
    fig.savefig(os.path.join(os.getcwd(), 'model', 'draws', 'LSTM',
                             f"LSTM_approach_1 - Train and test loss - {datetime.today().strftime('%Y-%m-%d_%H-%M')}.png"))

    # Plot the model performance
    test_target = y_test.cpu().detach().numpy()
    test_predictions = []
    for i in tqdm(range(len(test_target)), desc='Preparing predictions'):
        input_ = X_test[i].reshape(1, hyperparams.T.value, D)
        p = model(input_)[0, 0].item()
        test_predictions.append(p)

    plot_len = len(test_predictions)
    plot_df = df[['timestamp', 'y']].copy(deep=True)
    plot_df = plot_df.iloc[-plot_len:]
    plot_df['pred'] = test_predictions
    plot_df.set_index('timestamp', inplace=True)

    _, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(plot_df['y'], label='ARPA pm25', linewidth=1)
    ax.plot(plot_df['pred'], label='Predicted pm25', linewidth=1)
    ax.set_xlabel('timestamp')
    ax.set_ylabel(r'$\mu g/m^3$')
    ax.set_title(f'LSTM Performance - {hyperparams.NUM_EPOCHS.value} epochs - T = {hyperparams.T.value}')
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(os.getcwd(), 'model', 'draws', 'LSTM',
                             f"LSTM_approach_1 - Performance - {datetime.today().strftime('%Y-%m-%d_%H-%M')}.png"))


main()
