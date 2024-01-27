import os

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, RandomSampler

from config.config_parser import ConfigParser
from model.ANN import PM25AnnDataset2, MyNeuralNetwork
from model.train.ANN_trainer import ANN_trainer
from model.train.hyperparams.ann_hyperparams import ANN_Hyperparameters
from utils.dataset_utils import DatasetUtils
from utils.tuning_utils import TuningUtils

# Define constants
START_DATE_BOARD = '2022-11-03'
END_DATE_BOARD = '2023-06-15'
RANDOM_STATE = 42
NUM_SPLIT = 4
NUM_FEATURES = 4  # pm25, temperature, humidity, pressure


def build_dataset(cfg: ConfigParser) -> tuple:
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
    X = df.loc[:, df.columns != "y"]
    y = df['y']
    return X.values, y.values


def prepare_train_split(X_train, y_train, batch_loader_size: int = 2) -> DataLoader:
    # Convert to 2D PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
    # Convert into my dataset custom class
    train_dataset = PM25AnnDataset2(X_train, y_train)
    # Use data-loader in order to have batches
    train_loader = DataLoader(train_dataset, batch_size=batch_loader_size, shuffle=False, num_workers=0,
                              sampler=RandomSampler(train_dataset))
    return train_loader


def prepare_test_split(X_test, y_test, batch_loader_size: int = 2) -> DataLoader:
    # Convert to 2D PyTorch tensors
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)
    # Convert into my dataset custom class
    test_dataset = PM25AnnDataset2(X_test, y_test)
    # Use data-loader in order to have batches
    test_loader = DataLoader(test_dataset, batch_size=batch_loader_size, shuffle=False, num_workers=0,
                             sampler=RandomSampler(test_dataset))
    return test_loader


def main() -> None:
    print('Start ANN #2 cross-validation')
    # Get project configurations
    cfg = ConfigParser()
    # Set the best hyperparams combination
    hyperparams = ANN_Hyperparameters(
        {
            'TRAIN_SIZE': 0.75,
            'LEARNING_RATE': 0.0001,
            'OPTIMIZER': 'sgd',
            'CRITERION': 'l1',
            'HIDDEN_SIZE': 30,
            'HIDDEN_SIZE_2': 15,
            'HIDDEN_SIZE_3': 5,
            'NUM_EPOCHS': 200,
            'BATCH_SIZE': 10
        }
    )
    # Prepare dataset
    X, y = build_dataset(cfg)
    # Prepare cross-validation
    kf = KFold(n_splits=NUM_SPLIT, shuffle=True, random_state=RANDOM_STATE)
    # Loop fo each fold
    for i, (train_index, test_index) in enumerate(kf.split(X, y)):
        name = f"ANN#2_CV_FOLD{i + 1}"
        # Prepare the final dataset
        train_loader = prepare_train_split(X[train_index], y[train_index], hyperparams['BATCH_SIZE'])
        test_loader = prepare_test_split(X[test_index], y[test_index], hyperparams['BATCH_SIZE'])
        # Instantiate the model
        model = MyNeuralNetwork(NUM_FEATURES, 1, hyperparams['HIDDEN_SIZE'], hyperparams['HIDDEN_SIZE_2'],
                                hyperparams['HIDDEN_SIZE_3'])
        # Get the optimizer and criterion
        optimizer = TuningUtils.choose_optimizer(hyperparams.hyperparameters, model)
        criterion = TuningUtils.choose_criterion(hyperparams.hyperparameters)
        # Instantiate the trainer
        trainer = ANN_trainer(model, name=name, hyperparameters=hyperparams, optimizer=optimizer, criterion=criterion)
        train_losses, test_losses = trainer.train_loader(train_loader, test_loader)
        # Save hparams result on Tensorboard
        trainer.writer.add_hparams(hyperparams.hyperparameters,
                                   {'loss/train': train_losses.min(), 'loss/test': test_losses.min()})
        # Plot the train loss and test loss per iteration
        fig = trainer.draw_train_test_loss(train_losses, test_losses)
        trainer.save_image(f'{name} - Train and test loss', fig)

    print('End of ANN #2 cross-validation')


if __name__ == '__main__':
    main()
