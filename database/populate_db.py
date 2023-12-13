import json
import os

from database.interface.db_interface import DBInterface, MySQLUrl
from database.model.board import Board
from database.model.logical_sensor import LogicalSensor
from database.model.unit_of_measure import UnitOfMeasure
from database.service.boards_service import BoardService
from database.service.logical_sensors_service import LogicalSensorService
from database.service.unit_of_measure_service import UnitOfMeasureService

ROOT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


########
# THIS IS A GARBAGE SCRIPT USED TO CRATE A MOCKUP DB
########

def main():
    # connect to db
    # url = SQLiteUrl("test_new.db").get_url()
    url = MySQLUrl("weather_station_v2", "weather_station_local", "5NcmIt%Gk6&X6VH8dP", "127.0.0.1", 3306).get_url()
    db = DBInterface(url, echo=True)

    # create database
    db.create_db()

    # create tables
    db.create_all()

    measures = {
        "pm25": "mu_g/m^3",
        "pm10": "mu_g/m^3",
        "rh": "%",
        "temp": "°C",
        "pres": "hpa",
        "gps_lat": "deg",
        "gps_lon": "deg"
    }

    # start transaction
    with db.Session() as session:
        # create unit of measure
        uom_to_id = {}
        for name, unit in measures.items():
            uom = UnitOfMeasure(measureName=name, unitOfMeasure=unit)
            uom = UnitOfMeasureService.create(session, uom)
            uom_to_id[name] = uom.unitId

        # create boards
        with open(os.path.join(ROOT_DIR, "resources", "board.json")) as f:
            board_confs = json.load(f)
        for conf in board_confs:
            board = Board(boardId=conf["board_id"])
            print(board.boardId)
            BoardService.create_board(session, board)

        # create sensors
        for conf in board_confs:
            keys = set(conf.keys())
            keys.remove("board_id")
            board_id = conf["board_id"]
            for k in keys:
                m_id = uom_to_id[k]
                sens = conf[k]
                print(m_id, sens)
                for s in sens:
                    ls = LogicalSensor(sensorId=s, unitId=m_id, boardId=board_id)
                    LogicalSensorService.create(session, ls)

        session.commit()
