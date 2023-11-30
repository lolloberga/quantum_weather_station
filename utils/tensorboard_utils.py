import torch
from matplotlib import pyplot as plt


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
