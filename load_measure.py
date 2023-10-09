from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from dataset.interface.db_interface import DBInterface, MySQLUrl
from dataset.model.measure_temporary import MeasureTemporary
from config import ConfigParser
import os
import json
import csv

ROOT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

def main():
    # get configurations
    cfg = ConfigParser()
    
    RESOURCE_PATH = os.path.join(ROOT_DIR, cfg['resouces']['path']) if cfg['resouces']['path'] is not None else os.path.join(ROOT_DIR, "resources")
    DATASET_PATH = os.path.join(ROOT_DIR, cfg['dataset']['path'])
    FOLDER_NAME_PREFIX = cfg['dataset']['folder_name_prefix'] if cfg['dataset']['folder_name_prefix'] is not None else 'board_'
    SENSOR_NAME_PREFIX = cfg['dataset']['sensor_name_prefix'] if cfg['dataset']['sensor_name_prefix'] is not None else 's'
    BOARD_CONFIG_FILE = os.path.join(RESOURCE_PATH, cfg['dataset']['board_config_name']) if cfg['dataset']['board_config_name'] is not None else os.path.join(RESOURCE_PATH, 'board.json')

    if not os.path.exists(DATASET_PATH) or os.path.isfile(DATASET_PATH):
        print('Download path not available')
        exit(1)
    if not os.path.exists(BOARD_CONFIG_FILE) or not os.path.isfile(BOARD_CONFIG_FILE):
        print('Board config file not available')
        exit(1)


    # connect to db
    url = MySQLUrl(cfg['database']['db_name'], cfg['database']['user'], cfg['database']['password'], cfg['database']['url'], cfg['database']['port']).get_url()
    db = DBInterface(url, echo=True)

    # start transaction
    with db.Session() as session:
        for filename in os.scandir(DATASET_PATH):
            if not filename.is_file() and filename.name.startswith(FOLDER_NAME_PREFIX):
                board_number = int(filename.name.split(FOLDER_NAME_PREFIX)[1])
                board = _find_board_by_id(BOARD_CONFIG_FILE, board_number)
                print(filename.path, filename.name, board_number)
                print(board)
                if board is None:
                    continue
                
                # open CSV file
                for csv_file in os.scandir(filename.path):
                    if csv_file.is_file() and csv_file.name.endswith('.csv') and csv_file.name.startswith(SENSOR_NAME_PREFIX):
                        sensor_id = os.path.splitext(csv_file.name)[0].split(SENSOR_NAME_PREFIX)[1]
                        print(sensor_id)
                        try:
                            sensor_id = int(sensor_id)
                            _load_to_temp_table(session, csv_file.path, sensor_id, skip_first=True)
                        except ValueError:
                            continue
                session.commit()


def _load_to_temp_table(session: Session, csv_file: str, sensor_id: id, skip_first:bool = True) -> None:
    with open(csv_file) as f:
        csv_reader = csv.reader(f, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0 and skip_first:
                line_count += 1
            else:
                if len(row) >= 2:
                    timestamp = row[0]
                    value = float(row[1])
                    mt = MeasureTemporary(sensorId=sensor_id, timestamp=timestamp, data=value)
                    session.add(mt)

    
def _find_board_by_id(board_config_file: str, id: int) -> dict:
    with open(board_config_file) as f:
        board_confs = json.load(f)
        for conf in board_confs:
            if 'board_id' in conf and conf['board_id'] == id:
                return conf
    return None

if __name__ == "__main__":
    main()