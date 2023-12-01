import os
from functools import partial

import numpy as np
import pandas as pd
import torch
from filelock import FileLock
from matplotlib import pyplot as plt
from ray.tune import CLIReporter
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm

from config import ConfigParser
from model.ANN import PM25AnnDataset2, MyNeuralNetwork
from model.train.ANN_trainer import ANN_trainer
from model.train.base.hyperparameters import Hyperparameters
from model.train.base.trainer import Trainer
from model.train.hyperparams.ann_hyperparams import ANN_Hyperparameters
from utils.dataset_utils import DatasetUtils

from ray import tune, train
from ray.tune.schedulers import ASHAScheduler

# Define constants
START_DATE_BOARD = '2022-11-03'
END_DATE_BOARD = '2023-06-15'
RANDOM_STATE = 42
TRAIN_SIZE = 0.7
BATCH_LOADER_SIZE = 2
# Get global project configurations
cfg = ConfigParser()
DATASET_PATH = os.path.join(cfg.consts['RESOURCE_PATH'], 'dataset')


def build_dataset_2(cfg: ConfigParser, batch_loader_size: int = BATCH_LOADER_SIZE) -> tuple:
    df_sensors = pd.read_csv(
        os.path.join(DATASET_PATH, 'unique_timeseries_by_median_minutes.csv'))
    df_sensors.timestamp = pd.to_datetime(df_sensors.timestamp)
    df_sensors.timestamp += pd.Timedelta(minutes=1)
    df_arpa = DatasetUtils.build_arpa_dataset(os.path.join(DATASET_PATH, 'arpa',
                                                           'Dati PM10_PM2.5_2020-2022.csv')
                                              , os.path.join(DATASET_PATH, 'arpa',
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
    columns = ["tm{}".format(i) for i in range(1, 61)]
    columns.insert(0, 'arpa')
    df = pd.DataFrame(columns=columns)

    df_sensors.reset_index(inplace=True, drop=True)
    for i, arpa in enumerate(df_arpa['pm25']):
        row = df_sensors['data'][i * 60: (i + 1) * 60].values
        row = np.append(arpa, row)
        df.loc[len(df)] = row.tolist()

    X = df.loc[:, df.columns != "arpa"]
    y = df['arpa']
    X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, train_size=TRAIN_SIZE,
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

    with FileLock(os.path.expanduser(os.path.join(DATASET_PATH, '.data.lock'))):
        # Use data-loader in order to have batches
        train_loader = DataLoader(train_dataset, batch_size=batch_loader_size, shuffle=False, num_workers=0,
                                  sampler=RandomSampler(train_dataset))
        test_loader = DataLoader(test_dataset, batch_size=batch_loader_size, shuffle=False, num_workers=0,
                                 sampler=RandomSampler(test_dataset))
        return train_loader, test_loader, df_arpa


def plot_performance(model: nn.Module, test_loader: DataLoader, df_arpa: pd.DataFrame, trainer: Trainer,
                     name: str) -> None:
    test_target = test_loader.dataset.y.cpu().detach().numpy()
    test_predictions = []
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(len(test_target)), desc='Preparing predictions'):
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
    ax.set_title(name)
    ax.legend(loc='upper right')
    fig.tight_layout()
    trainer.save_image('ANN - Performance', fig)


def train_hyperparams(hyperparams) -> None:
    # Get project configurations
    cfg = ConfigParser()
    # Get default hyperparams and customize them
    default_hyperparams = ANN_Hyperparameters()
    # Hyperparameters.change_value(default_hyperparams.hyperparameters.NUM_EPOCHS.value, hyperparams['epochs'])
    # Prepare dataset
    train_loader, test_loader, df_arpa = build_dataset_2(cfg, batch_loader_size=hyperparams['batch_size'])
    # Instantiate the model
    # model = MyNeuralNetwork(60, 1, hyperparams['l1'], hyperparams['l2'], hyperparams['l3'])
    model = MyNeuralNetwork(60, 1, hyperparams['l1'])
    trainer = ANN_trainer(model, name='ANN_tuning_1', hyperparameters=default_hyperparams)

    checkpoint: train.Checkpoint = train.get_checkpoint()
    if checkpoint:
        with checkpoint.as_directory() as checkpoint_dir:
            print('Checkpoint dir', checkpoint_dir)
            model_state, optimizer_state = torch.load(
                os.path.join(checkpoint_dir, "checkpoint"))
            trainer.model.load_state_dict(model_state)
            trainer.get_optimizer().load_state_dict(optimizer_state)

    train_losses, test_losses = trainer.train_loader(train_loader, test_loader, use_ray_tune=True)


def main(num_samples=10, max_num_epochs=10, gpus_per_trial=2):
    print('Start ANN hyperparameters tuning')
    # Get configuration space
    config = {
        # "l1": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
        "l1": tune.choice([128]),
        # "l2": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
        "batch_size": tune.choice([2]),
        "epochs": tune.choice([1])
    }
    # Setting up the tuning process
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)
    reporter = CLIReporter(
        # parameter_columns=["l1", "l2", "lr", "batch_size"],
        metric_columns=["loss", "training_iteration"])
    # Run the tuning
    result = tune.run(
        partial(train_hyperparams),
        storage_path=os.path.join(os.getcwd(), 'runs'),
        name='ANN_Tuning',
        local_dir=os.path.join(os.getcwd(), 'runs'),
        resources_per_trial={"cpu": 2, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter)

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))


if __name__ == "__main__":
    os.environ["TUNE_PLACEMENT_GROUP_AUTO_DISABLED"] = "1"
    main(num_samples=2, max_num_epochs=10, gpus_per_trial=0)
