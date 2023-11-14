from config import ConfigParser
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

# Define constants
torch.manual_seed(1)
RANDOM_STATE = 42
TRAIN_SIZE = 0.8
START_DATE_BOARD = '2022-11-03'
END_DATE_BOARD = '2023-06-15'
T = 5  # Number of hours to look while predicting

# Define the hyperparameters
HIDDEN_SIZE = 512
OUTPUT_SIZE = 1
LEARNING_RATE = 0.01
NUM_EPOCHS = 5


class MyLSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(MyLSTM, self).__init__()
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
    model = MyLSTM(D, HIDDEN_SIZE, 2, OUTPUT_SIZE)
    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=1e-4)
    train_losses = np.zeros(NUM_EPOCHS)
    test_losses = np.zeros(NUM_EPOCHS)

    # Train the model
    writer = SummaryWriter(os.path.join(os.getcwd(), 'runs', f"lstm_approach_1 - {datetime.today().strftime('%Y-%m-%d %H:%M')}"))
    for epoch in tqdm(range(NUM_EPOCHS), desc='Train the model'):
        optimizer.zero_grad()

        # Forward pass
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        writer.add_scalar("Loss/train", loss, epoch)

        # Backpropagation
        loss.backward()
        optimizer.step()

        # Train loss
        train_losses[epoch] = loss.item()

        # Test loss
        test_outputs = model(X_test)
        test_loss = criterion(test_outputs, y_test)
        writer.add_scalar("Loss/test", test_loss, epoch)
        test_losses[epoch] = test_loss.item()

        # if (epoch + 1) % 50 == 0:
        #    print(
        #        f'At epoch {epoch + 1} of {NUM_EPOCHS}, Train Loss: {loss.item():.3f}, Test Loss: {test_loss.item():.3f}')

    # Save the model at the end of the training (for future inference)
    torch.save(model.state_dict(), os.path.join(os.getcwd(), 'model', 'checkpoints', 'lstm_approach_1.pt'))
    writer.flush()
    writer.close()

    # Plot the train loss and test loss per iteration
    _, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(train_losses, label='Train loss')
    ax.set_xlabel('epoch no')
    ax.set_ylabel('loss')
    ax.set_title(f'Train loss at each iteration - {NUM_EPOCHS} epochs - T = {T}')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(os.getcwd(), 'model', 'draws', 'LSTM', f'LSTM_approach_1 - Train and test loss.png'))

    # Plot the model performance
    test_target = y_test.cpu().detach().numpy()
    test_predictions = []
    for i in tqdm(range(len(test_target)), desc='Preparing predictions'):
        input_ = X_test[i].reshape(1, T, D)
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
    ax.set_title(f'LSTM Performance - {NUM_EPOCHS} epochs - T = {T}')
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(os.getcwd(), 'model', 'draws', 'LSTM', f'LSTM_approach_1 - Train and test loss.png'))


main()
