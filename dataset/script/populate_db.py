from db_lib.db_interface import DBInterface, SQLiteUrl, MySQLUrl
from db_lib.services import BoardService, LogicalSensorService, UnitOfMeasureService
from db_lib.models import Board, LogicalSensor, UnitOfMeasure
import json
import sshtunnel
from sshtunnel import SSHTunnelForwarder
import logging

from sqlalchemy import create_engine


########
# THIS IS A GARBAGE SCRIPT USED TO CRATE A MOCKUP DB
########

def open_ssh_tunnel(verbose=False):
    """Open an SSH tunnel and connect using a username and password.

    :param verbose: Set to True to show logging
    :return tunnel: Global SSH tunnel connection
    """
    if verbose:
        sshtunnel.DEFAULT_LOGLEVEL = logging.DEBUG

    global tunnel
    tunnel = SSHTunnelForwarder(
        ("baraddur.polito.it", 22),
        ssh_username="lorenzo",
        ssh_password="B3rg4d4n0!",
        remote_bind_address=('127.0.0.1', 3306)
    )
    tunnel.start()


def main():

    open_ssh_tunnel(True)

    try:
        # connect to db
        # url = SQLiteUrl("test_new.db").get_url()
        # url = MySQLUrl("weather_station_v2", "weather_station", "5NcmIt%Gk6&X6VH8dP", "geonosis.polito.it", tunnel.local_bind_port).get_url()
        url = MySQLUrl("weather_station_v2", "weather_station", "5NcmIt%Gk6&X6VH8dP", "192.168.16.117",
                       tunnel.local_bind_port).get_url()
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
            with open("board.json") as f:
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
    except Exception as e:
        print(e)
        tunnel.close()


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
           