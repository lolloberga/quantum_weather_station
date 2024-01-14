import pandas as pd
import torch
from matplotlib import pyplot as plt
from tbparse import SummaryReader


class TensorboardUtils:

    @staticmethod
    def draw_prediction_tensorboard(prediction: torch.Tensor, actual: torch.Tensor, epoch: int) -> plt.Figure:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        ax.plot(actual.detach().numpy().T[0], label='ARPA pm25', linewidth=1)
        ax.plot(prediction.detach().numpy().T[0], label='Predicted pm25', linewidth=1)
        ax.set_xlabel('timestamp')
        ax.set_ylabel(r'$\mu g/m^3$')
        # ax.set_title(f'LSTM Performance - At epoch {epoch + 1}')
        ax.legend(loc='upper right')
        fig.tight_layout()
        return fig

    @staticmethod
    def read_hyperparameters(writer_path: str) -> pd.DataFrame:
        reader = SummaryReader(writer_path)
        return reader.hparams


def main(log_dir: str) -> None:
    hp = TensorboardUtils.read_hyperparameters(log_dir)
    print(hp)
