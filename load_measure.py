from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from dataset.interface.db_interface import DBInterface, MySQLUrl
from dataset.model.measure_temporary import MeasureTemporary
from dataset.script.download_datatest import DownloadDataset
from config import ConfigParser
from tqdm import tqdm
import os
import json
import csv

ROOT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MEASURES = {
    "pm25": "mu_g/m^3",
    "pm10": "mu_g/m^3",
    "rh" : "%",
    "temp": "°C",
    "pres": "hpa",
    "gps_lat": "deg",
    "gps_lon": "deg"
}

def main():
    # get configurations
    cfg = ConfigParser()
    # get constants
    DATASET_PATH = cfg.consts['DATASET_PATH']
    FOLDER_NAME_PREFIX = cfg.consts['FOLDER_NAME_PREFIX']
    SENSOR_NAME_PREFIX = cfg.consts['SENSOR_NAME_PREFIX']
    BOARD_CONFIG_FILE = cfg.consts['BOARD_CONFIG_FILE']

    if not os.path.exists(DATASET_PATH) or os.path.isfile(DATASET_PATH):
        print('Download path not available')
        exit(1)
    if not os.path.exists(BOARD_CONFIG_FILE) or not os.path.isfile(BOARD_CONFIG_FILE):
        print('Board config file not available')
        exit(1)

    # download dataset
    dd = DownloadDataset()
    dd.download_with_map_file()

    # connect to db
    url = MySQLUrl(cfg['database']['db_name'], cfg['database']['user'], cfg['database']['password'], cfg['database']['url'], cfg['database']['port']).get_url()
    db = DBInterface(url, echo=True)

    # start transaction
    with db.Session() as session:
        for filename in tqdm(os.scandir(DATASET_PATH), desc=f'Uploading dataset...'):
            if not filename.is_file() and filename.name.startswith(FOLDER_NAME_PREFIX):
                board_number = int(filename.name.split(FOLDER_NAME_PREFIX)[1])
                board = _find_board_by_id(BOARD_CONFIG_FILE, board_number)
                #print(filename.path, board_number, board)
                if board is None:
                    continue
                
                # open CSV file
                for csv_file in os.scandir(filename.path):
                    if csv_file.is_file() and csv_file.name.endswith('.csv') and csv_file.name.startswith(SENSOR_NAME_PREFIX):
                        sensor_id = os.path.splitext(csv_file.name)[0].split(SENSOR_NAME_PREFIX)[1]
                        try:
                            sensor_id = int(sensor_id)
                            # check if sensor_id is inside the board
                            measure = _check_sensor_id_into_board(sensor_id, board)
                            if measure is not None:
                                # load into temporary table
                                print(f'Upload sensor {sensor_id}({measure})')
                                _load_to_temp_table(session, csv_file.path, sensor_id, skip_first=True)
                                session.commit()
                        except ValueError:
                            continue
                session.commit()
    
    print('Finish push into temporary measures')


def _load_to_temp_table(session: Session, csv_file: str, sensor_id: id, skip_first:bool = True) -> None:
    with open(csv_file) as f:
        csv_reader = csv.reader(f, delimiter=',')
        line_count = 0
        for row in tqdm(csv_reader, desc=f'Upload rows of sensor {sensor_id}'):
            line_count += 1
            if line_count == 1 and skip_first:
                continue
            else:
                if len(row) >= 2:
                    timestamp = row[0]
                    value = float(row[1])
                    mt = MeasureTemporary(sensorId=sensor_id, timestamp=timestamp, data=value)
                    session.add(mt)

        print(f'- Loaded {line_count} rows')

    
def _find_board_by_id(board_config_file: str, id: int) -> dict:
    with open(board_config_file) as f:
        board_confs = json.load(f)
        for conf in board_confs:
            if 'board_id' in conf and conf['board_id'] == id:
                return conf
    return None

def _check_sensor_id_into_board(sensor_id: int, board: dict) -> str:
    for measure in MEASURES.keys():
        if measure in board:
            if sensor_id in board[measure]:
                return measure
    return None

if __name__ == "__main__":
    main()