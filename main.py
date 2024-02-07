import argparse
from database.load_measure import main as load_measure_table
from database.populate_db import main as populate_db
from model.ANN import main as ann_model
from model.LSTM import main as lstm_model
from model.QLSTM import main as qlstm_model
from model.VQRNonLinear import main as vqr_nonlinear_model
from model.VQRLinear import main as vqr_linear_model
from hyperparams_tuning.ANN_tuning import main as ann_tuning
from hyperparams_tuning.LSTM_tuning import main as lstm_tuning
from hyperparams_tuning.LSTM_qlstm_tuning import main as lstm_qlstm_tuning
from hyperparams_tuning.VQRNonLinear_tuning import main as vqr_nonlinear_tuning
from hyperparams_tuning.VQRLinear_tuning import main as vqr_linear_tuning
from hyperparams_tuning.QLSTM_basic_tuning import main as qlstm_tuning
from hyperparams_tuning.QLSTM_strongly_tuning import main as qlstm_strongly_tuning
from improvements.LSTM_improvement1 import main as lstm_1
from improvements.LSTM_improvement2 import main as lstm_2
from improvements.ANN_improvement1 import main as ann_1
from improvements.ANN_improvement2 import main as ann_2
from improvements.VQRLinear_improvement1 import main as vqr_linear_test1
from improvements.VQRNonLinear_improvement1 import main as vqr_nonlinear_test1
from cross_validation.VQR_Linear_CV import main as vqr_linear_cv
from cross_validation.VQR_NonLinear_CV import main as vqr_nonlinear_cv
from utils.tensorboard_utils import TensorboardUtils
from cross_validation.LSTM_CV import main as lstm_cv

ACTIONS = ['LOAD_MEASURE_TABLE', 'POPULATE_DB', 'ANN_MODEL', 'LSTM_MODEL', 'ANN_TUNING', 'LSTM_TUNING', 'LSTM_#1',
           'ANN_#1', 'ANN_#2', 'VQR_NONLINEAR_MODEL', 'VQR_LINEAR_MODEL', 'VQR_NONLINEAR_TUNING', 'VQR_LINEAR_TUNING',
           'TB_READ_HP', 'LSTM_CV', 'VQR_LINEAR_TEST#1', 'VQR_NONLINEAR_TEST#1', 'VQR_LINEAR_CV', 'VQR_NONLINEAR_CV',
           'QLSTM_MODEL', 'QLSTM_BASIC_TUNING', 'QLSTM_STRONGLY_TUNING', 'LSTM_#2', 'LSTM_QLSTM_TUNING']


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
    elif args.action == 'QLSTM_MODEL':
        qlstm_model()
    elif args.action == 'ANN_TUNING':
        ann_tuning()
    elif args.action == 'LSTM_TUNING':
        lstm_tuning()
    elif args.action == 'LSTM_QLSTM_TUNING':
        lstm_qlstm_tuning()
    elif args.action == 'QLSTM_BASIC_TUNING':
        qlstm_tuning()
    elif args.action == 'QLSTM_STRONGLY_TUNING':
        qlstm_strongly_tuning()
    elif args.action == 'LSTM_#1':
        lstm_1()
    elif args.action == 'LSTM_#2':
        lstm_2()
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
    elif args.action == 'VQR_LINEAR_CV':
        vqr_linear_cv()
    elif args.action == 'VQR_NONLINEAR_CV':
        vqr_nonlinear_cv()
    else:
        print('There was an error during execute your program.')
        parser.print_help()
        exit(-1)