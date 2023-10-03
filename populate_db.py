from dataset.interface.db_interface import DBInterface, SQLiteUrl, MySQLUrl
from dataset.service.boards_service import BoardService
from dataset.service.logical_sensors_service import LogicalSensorService
from dataset.service.unit_of_measure_service import UnitOfMeasureService
from dataset.model.board import Board
from dataset.model.logical_sensor import LogicalSensor
from dataset.model.unit_of_measure import UnitOfMeasure
import json

from sqlalchemy import create_engine

import os
ROOT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


########
# THIS IS A GARBAGE SCRIPT USED TO CRATE A MOCKUP DB
########

def main():

    # connect to db
    url = MySQLUrl("weather_station_v2", "weather_station_local", "5NcmIt%Gk6&X6VH8dP", "localhost",
                    3306).get_url()
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


if __name__ == "__main__":
    main()

# This was used to clean the json
# for line in board_confs:
#     if "gps(lt-lg)" in line:
#         lat_sens = line["gps(lt-lg)"][0]
#         lon_sens = line["gps(lt-lg)"][1]
#         line.pop("gps(lt-lg)")
#         line["gps_lat"] = [lat_sens]
#         line["gps_lon"] = [lon_sens]
#         print(line)
           