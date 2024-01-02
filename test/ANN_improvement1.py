import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, RandomSampler

from config.config_parser import ConfigParser
from model.ANN import PM25AnnDataset2, MyNeuralNetwork
from model.train.LSTM_trainer import LSTM_trainer
from model.train.hyperparams.ann_hyperparams import ANN_Hyperparameters
from utils.dataset_utils import DatasetUtils
from utils.tuning_utils import TuningUtils

"""
    THIS PYTHON SCRIPT IS RELATED TO TEST A PARTICULAR COMBINATION OF DATASET AND HYPERPARAMS.
    YOU CAN SEE THE FULL DESCRIPTION OF THE TEST HERE: https://trello.com/c/nUTo7oBe/19-ann-improvement-1
"""

# Define constants
START_DATE_BOARD = '2022-11-03'
END_DATE_BOARD = '2023-06-15'
RANDOM_STATE = 42
NAME = f'ANN_all_feats_{datetime.today().strftime("%Y%m%d_%H%M%S")}'
NUM_FEATURES = 4 # pm25, temperature, humidity, pressure


def build_dataset_2(cfg: ConfigParser, batch_loader_size: int = 2, train_size: float = 0.7) -> tuple:
    df_sensors = pd.read_csv(
        os.path.join(cfg.consts['DATASET_PATH'], 'unique_timeseries_by_median_minutes.csv'))
    df_sensors.timestamp = pd.to_datetime(df_sensors.timestamp)
    df_sensors.timestamp += pd.Timedelta(minutes=60)
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
    columns_pm25 = ["pm{}".format(i) for i in range(1, 61)]
    columns_temp = ["tm{}".format(i) for i in range(1, 61)]
    columns_pres = ["pr{}".format(i) for i in range(1, 61)]
    columns_rh = ["rh{}".format(i) for i in range(1, 61)]
    columns = columns_pm25 + columns_temp + columns_pres + columns_rh
    columns.insert(0, 'arpa')
    df = pd.DataFrame(columns=columns)

    df_sensors.reset_index(inplace=True, drop=True)
    for i, arpa in enumerate(df_arpa['pm25']):
        row_pm25 = df_sensors['pm25'][i * 60: (i + 1) * 60].values
        row_temp = df_sensors['temp'][i * 60: (i + 1) * 60].values
        row_pres = df_sensors['pres'][i * 60: (i + 1) * 60].values
        row_rh = df_sensors['rh'][i * 60: (i + 1) * 60].values
        row = np.concatenate([row_pm25, row_temp, row_pres, row_rh])
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


def main() -> None:
    print(f'Start {NAME}')
    # Get project configurations
    cfg = ConfigParser()
    hyperparams = ANN_Hyperparameters(
        {
            'TRAIN_SIZE': 0.75,
            'LEARNING_RATE': 0.0001,
            'OPTIMIZER': 'sgd',
            'CRITERION': 'l1',
            'HIDDEN_SIZE': 650,
            'HIDDEN_SIZE_2': 90,
            'HIDDEN_SIZE_3': 30,
            'NUM_EPOCHS': 250,
            'MOMENTUM': 0.9,
            'WEIGHT_DECAY': 0,
            'BATCH_SIZE': 2
        }
    )
    # Prepare dataset
    train_loader, test_loader, df_arpa = build_dataset_2(cfg, batch_loader_size=hyperparams['BATCH_SIZE'],
                                                         train_size=hyperparams['TRAIN_SIZE'])
    # Instantiate the model
    model = MyNeuralNetwork(60 * NUM_FEATURES, 1, hyperparams['HIDDEN_SIZE'], hyperparams['HIDDEN_SIZE_2'],
                            hyperparams['HIDDEN_SIZE_3'])
    # Get the correct optimizer and criterion
    optimizer = TuningUtils.choose_optimizer(hyperparams.hyperparameters, model)
    criterion = TuningUtils.choose_criterion(hyperparams.hyperparameters)
    # Instantiate the trainer
    trainer = LSTM_trainer(model, name=NAME, hyperparameters=hyperparams, optimizer=optimizer, criterion=criterion)
    train_losses, test_losses = trainer.train_loader(train_loader, test_loader)
    # Plot the train loss and test loss per iteration
    fig = trainer.draw_train_test_loss(train_losses, test_losses)
    trainer.save_image(f'{NAME} - Train and test loss', fig)
    # Plot the model performance
    test_target = test_loader.dataset.y.cpu().detach().numpy()
    test_predictions = []
    model.eval()
    with torch.no_grad():
        for i in range(len(test_target)):
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
    ax.set_title(f'ANN Performance - {hyperparams["NUM_EPOCHS"]} epochs')
    ax.legend(loc='lower right')
    fig.tight_layout()
    trainer.save_image(f'{NAME} - Performance', fig)
    print(f'End {NAME}')
