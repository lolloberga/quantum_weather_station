import argparse
from database.load_measure import main as load_measure_table
from database.populate_db import main as populate_db
from model.ANN import main as ann_model
from model.LSTM import main as lstm_model
from model.VQRNonLinear import main as vqr_nonlinear_model
from model.VQRLinear import main as vqr_linear_model
from hyperparams_tuning.ANN_tuning import main as ann_tuning
from hyperparams_tuning.LSTM_tuning import main as lstm_tuning
from hyperparams_tuning.VQRNonLinear_tuning import main as vqr_nonlinear_tuning
from hyperparams_tuning.VQRLinear_tuning import main as vqr_linear_tuning
from test.LSTM_improvement1 import main as lstm_1
from test.ANN_improvement1 import main as ann_1
from test.ANN_improvement2 import main as ann_2
from test.VQRLinear_improvement1 import main as vqr_linear_test1
from test.VQRNonLinear_improvement1 import main as vqr_nonlinear_test1
from utils.tensorboard_utils import TensorboardUtils
from cross_validation.LSTM_CV import main as lstm_cv

ACTIONS = ['LOAD_MEASURE_TABLE', 'POPULATE_DB', 'ANN_MODEL', 'LSTM_MODEL', 'ANN_TUNING', 'LSTM_TUNING', 'LSTM_#1',
           'ANN_#1', 'ANN_#2', 'VQR_NONLINEAR_MODEL', 'VQR_LINEAR_MODEL', 'VQR_NONLINEAR_TUNING', 'VQR_LINEAR_TUNING',
           'TB_READ_HP', 'LSTM_CV']


def set_mandatory_args(parser: argparse.ArgumentParser):
    parser.add_argument("-a", "--action", type=str, help="The action that you can execute", required=True,
                        choices=ACTIONS)
    parser.add_argument("-tb", "--tensorboard-folder", type=str, help="The TB folder used during the run",
                        required=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantum Weather Station - PoliTo project")
    set_mandatory_args(parser)

    args = parser.parse_args()
    if args.action not in ACTIONS:
        print(f'Action {args.action} is not a valid action.')
        parser.print_help()
        exit(-1)

    if args.action == 'LOAD_MEASURE_TABLE':
        load_measure_table()
    elif args.action == 'POPULATE_DB':
        populate_db()
    elif args.action == 'TB_READ_HP' and args.tensorboard_folder is not None:
        hp = TensorboardUtils.read_hyperparameters(args.tensorboard_folder)
        print(hp)
    elif args.action == 'ANN_MODEL':
        ann_model()
    elif args.action == 'LSTM_MODEL':
        lstm_model()
    elif args.action == 'ANN_TUNING':
        ann_tuning()
    elif args.action == 'LSTM_TUNING':
        lstm_tuning()
    elif args.action == 'LSTM_#1':
        lstm_1()
    elif args.action == 'ANN_#1':
        ann_1()
    elif args.action == 'ANN_#2':
        ann_2()
    elif args.action == 'VQR_NONLINEAR_MODEL':
        vqr_nonlinear_model()
    elif args.action == 'VQR_LINEAR_MODEL':
        vqr_linear_model()
    elif args.action == 'VQR_NONLINEAR_TUNING':
        vqr_nonlinear_tuning()
    elif args.action == 'VQR_LINEAR_TUNING':
        vqr_linear_tuning()
    elif args.action == 'LSTM_CV':
        lstm_cv()
    elif args.action == 'VQR_LINEAR_TEST#1':
        vqr_linear_test1()
    elif args.action == 'VQR_NONLINEAR_TEST#1':
        vqr_nonlinear_test1()
    else:
        print('There was an error during execute your program.')
        parser.print_help()
        exit(-1)