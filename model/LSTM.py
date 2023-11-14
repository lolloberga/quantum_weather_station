from config import ConfigParser
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, SequentialSampler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Define constants
torch.manual_seed(1)
RANDOM_STATE = 42
TRAIN_SIZE = 0.8
START_DATE_BOARD = '2022-11-03'
END_DATE_BOARD = '2023-06-15'
T = 5  # Number of timesteps to look while predicting

# Define the hyperparameters
INPUT_SIZE = HIDDEN_SIZE = 64  # a time window of 64-hour
OUTPUT_SIZE = 1
LEARNING_RATE = 0.01
NUM_EPOCHS = 400


class MyLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        batch_size = input.size(0)
        hidden = self.init_hidden(batch_size)
        lstm_out, hidden = self.lstm(input.view(len(input), batch_size, -1), hidden)
        output = self.fc(lstm_out[-1])
        return output

    def init_hidden(self, batch_size):
        return (torch.zeros(1, batch_size, self.hidden_size),
                torch.zeros(1, batch_size, self.hidden_size))


class MyLSTM_2(nn.Module):

    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(MyLSTM_2, self).__init__()
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
        h0 = torch.zeros(self.L, X.size(0), self.M)
        c0 = torch.zeros(self.L, X.size(0), self.M)

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


def build_dataset(cfg: ConfigParser) -> tuple:
    df_sensors = pd.read_csv(
        os.path.join(os.getcwd(), cfg.consts['DATASET_PATH'].replace('/', '', 1), 'unique_timeserie_by_median.csv'))
    df_sensors.timestamp = pd.to_datetime(df_sensors.timestamp)
    df_arpa = build_arpa_dataset(os.path.join(os.getcwd(), cfg.consts['DATASET_PATH'].replace('/', '', 1), 'arpa',
                                              'Dati PM10_PM2.5_2020-2022.csv')
                                 , os.path.join(os.getcwd(), cfg.consts['DATASET_PATH'].replace('/', '', 1), 'arpa',
                                                'Torino-Rubino_Polveri-sottili_2023-01-01_2023-06-30.csv'))

    df = df_sensors.merge(df_arpa, left_on=['timestamp'], right_on=['timestamp'])
    df.rename(columns={"data": "x", "pm25": "y"}, inplace=True)

    X_train, X_test, y_train, y_test = train_test_split(df.x.values, df.y.values,
                                                        train_size=TRAIN_SIZE, shuffle=False, random_state=RANDOM_STATE)
    # train_data = df.x.values[: int(len(df.x.values) * TRAIN_TEST_SIZE)]
    # test_data = df.x.values[int(len(df.x.values) * TRAIN_TEST_SIZE):]

    # Convert the time series data to PyTorch tensors
    X_train = torch.FloatTensor(X_train).unsqueeze(1)
    X_test = torch.FloatTensor(X_test).unsqueeze(1)
    y_train = torch.FloatTensor(y_train).unsqueeze(1)
    y_test = torch.FloatTensor(y_test).unsqueeze(1)
    # train_data = torch.FloatTensor(train_data).unsqueeze(1)
    # test_data = torch.FloatTensor(test_data).unsqueeze(1)
    return X_train, X_test, y_train, y_test


def build_dataset_2(cfg: ConfigParser) -> tuple:
    df_sensors = pd.read_csv(
        os.path.join(os.getcwd(), cfg.consts['DATASET_PATH'].replace('/', '', 1), 'unique_timeserie_by_median.csv'))
    df_sensors.timestamp = pd.to_datetime(df_sensors.timestamp)
    df_arpa = build_arpa_dataset(os.path.join(os.getcwd(), cfg.consts['DATASET_PATH'].replace('/', '', 1), 'arpa',
                                              'Dati PM10_PM2.5_2020-2022.csv')
                                 , os.path.join(os.getcwd(), cfg.consts['DATASET_PATH'].replace('/', '', 1), 'arpa',
                                                'Torino-Rubino_Polveri-sottili_2023-01-01_2023-06-30.csv'))

    df = df_sensors.merge(df_arpa, left_on=['timestamp'], right_on=['timestamp'])
    df.rename(columns={"data": "x", "pm25": "y"}, inplace=True)
    # Add month column and transform it into one-hot-encording
    df['month'] = df.timestamp.dt.month
    df = pd.get_dummies(df, columns = ['month'])

    input_data = df.drop(['timestamp', 'y'], axis=1)
    targets = df.y.values
    D = input_data.shape[1] # Dimensionality of the input
    N = len(input_data) - T

    train_size = int(len(input_data) * TRAIN_SIZE)
    # Preparing X_train and y_train
    X_train = np.zeros((train_size, T, D))
    y_train = np.zeros((train_size, 1))
    for t in range(train_size):
        X_train[t, :, :] = input_data[t:t+T]
        y_train[t] = (targets[t+T])

    # Preparing X_test and y_test
    X_test = np.zeros((N - train_size, T, D))
    y_test = np.zeros((N - train_size, 1))
    for i in range(N - train_size):
        t = i + train_size
        X_test[i, :, :] = input_data[t:t+T]
        y_test[i] = (targets[t+T])

    # Make inputs and targets
    X_train = torch.from_numpy(X_train.astype(np.float32))
    y_train = torch.from_numpy(y_train.astype(np.float32))
    X_test = torch.from_numpy(X_test.astype(np.float32))
    y_test = torch.from_numpy(y_test.astype(np.float32))

    return X_train, X_test, y_train, y_test, D


def main():
    # Get project configurations
    cfg = ConfigParser()
    # Prepare dataset
    X_train, X_test, y_train, y_test, D = build_dataset_2(cfg)

    # Instantiate the model
    model = MyLSTM_2(D, 512, 2, 1)
    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=1e-4)
    train_losses = np.zeros(NUM_EPOCHS)
    test_losses = np.zeros(NUM_EPOCHS)
    # Train the model
    for epoch in range(NUM_EPOCHS):
        optimizer.zero_grad()

        # Forward pass
        outputs = model(X_train)
        loss = criterion(outputs, y_train)

        # Backpropagation
        loss.backward()
        optimizer.step()

        #Train loss
        train_losses[epoch] = loss.item()

        # Test loss
        test_outputs = model(X_test)
        test_loss = criterion(test_outputs, y_test)
        test_losses[epoch] = test_loss.item()

        if (epoch + 1) % 50 == 0:
            print(f'At epoch {epoch+1} of {NUM_EPOCHS}, Train Loss: {loss.item():.3f}, Test Loss: {test_loss.item():.3f}')

    # Plot the train loss and test loss per iteration
    plt.plot(train_losses, label='train loss')
    plt.plot(test_losses, label='test loss')
    plt.xlabel('epoch no')
    plt.ylabel('loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(os.getcwd(), 'model', 'draws', f'LSTM_2 - Train and test loss.png'))



    '''
    # Instantiate the model
    model = MyLSTM(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # Create a data loader
    train_loader = DataLoader(X_train, batch_size=HIDDEN_SIZE, shuffle=False,
                              sampler=SequentialSampler(X_train), pin_memory=False)
    label_loader = DataLoader(y_train, batch_size=HIDDEN_SIZE, shuffle=False,
                              sampler=SequentialSampler(y_train), pin_memory=False)

    for epoch in range(NUM_EPOCHS):
        for i, (inputs, labels) in enumerate(zip(train_loader, label_loader)):
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, NUM_EPOCHS, i + 1, len(train_loader), loss.item()))
    '''


main()
