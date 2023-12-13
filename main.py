import argparse
from database.load_measure import main as load_measure_table
from database.populate_db import main as populate_db
from model.ANN import main as ann_model
from model.LSTM import main as lstm_model
from hyperparams_tuning.ANN_tuning import main as ann_tuning
from hyperparams_tuning.LSTM_tuning import main as lstm_tuning

ACTIONS = ['LOAD_MEASURE_TABLE', 'POPULATE_DB', 'ANN_MODEL', 'LSTM_MODEL', 'ANN_TUNING', 'LSTM_TUNING']


def set_mandatory_args(parser: argparse.ArgumentParser):
    parser.add_argument("-a", "--action", type=str, help="The action that you can execute", required=True,
                        choices=ACTIONS)


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
    elif args.action == 'ANN_MODEL':
        ann_model()
    elif args.action == 'LSTM_MODEL':
        lstm_model()
    elif args.action == 'ANN_TUNING':
        ann_tuning()
    elif args.action == 'LSTM_TUNING':
        lstm_tuning()

