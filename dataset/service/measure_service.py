from typing import List, Dict, Any
from Crypto.PublicKey import RSA
from Crypto.Hash import SHA256
from Crypto.Signature import pkcs1_15
import base64
from enum import IntEnum
from utils.db_utils import unix_to_datetime, object_as_dict

from dataset.model import (
    Board,
    BoardExperiment,
    LogicalSensor,
    Measure,
    PhysicalSensor
)

from dataset.view.five_min_avg_measure import FiveMinAvgMeasure
import struct
from sqlalchemy import func, or_, and_, types
from sqlalchemy.sql import expression
from sqlalchemy.orm import Session


from sqlalchemy import insert, select
from sqlalchemy.engine import Row


class MeasureStatus(IntEnum):
    VALID = 0
    NO_SIGNATURE = 1
    SIGN_NOT_VALID = 2
    FILENAME_INCONSISTENCY = 3


def verify_signature(key, dataFile, signFile):
    h = SHA256.new(base64.b64encode(bytearray(dataFile.stream.read())))
    signature = signFile.stream.read()
    pkcs1_15.new(key).verify(h, signature)


def bytes_to_measures(data) -> List[Dict[str, Any]]:
    data.stream.seek(0)
    measures = []
    while True:
        bytesRead = bytearray(data.stream.read(12))
        if not bytesRead:
            break
        timestamp = struct.unpack('!i', bytesRead[0:4])[0]
        sensorId = struct.unpack('!i', bytesRead[4:8])[0]
        measure = struct.unpack('!f', bytesRead[8:12])[0]
        measures.append((sensorId, timestamp, measure))
    return measures


class MeasureEncoder:
    @staticmethod
    def decode(bin_data: bytes) -> List[Dict[str, Any]]:
        index = 0
        measurements = []
        while(index < len(bin_data)):
            timestamp = unix_to_datetime(struct.unpack('!I', bin_data[index: index + 4])[0])
            index += 4
            sensorId = struct.unpack('!H', bin_data[index: index + 2])[0]
            index += 2
            sensor_type = chr(bin_data[index])
            index += 1
            if sensor_type == 'A' or sensor_type == 'G':
                measure = struct.unpack('!I', bin_data[index: index + 4])[0]
                if sensor_type == 'A':
                    measure = round(measure/10000, 4)
                else:
                    measure = round((measure/1000000)-180, 6)
                index += 4
            else:
                measure = struct.unpack('!H', bin_data[index: index + 2])[0]
                if sensor_type == 'T':
                    measure = round((measure/100)-40,2)
                elif sensor_type == 'H':
                    measure = round(measure/100,2)
                index += 2
            measurements.append({"sensorId":sensorId, "timestamp":timestamp, "data":measure})
        return measurements


class MeasureService:
    @staticmethod
    def add_measurements_bulk(session: Session, measures: List[Dict[str, Any]], ignore=False) -> None:
        # return if there are no measurements (None or empty)
        if not measures:
            return
        # this is a fast bulk insert (only one query, with no object creation)
        ins_query = insert(Measure)
        if ignore:
            ins_query = ins_query.prefix_with('IGNORE', dialect="mysql").prefix_with('OR IGNORE', dialect="sqlite")
        session.execute(ins_query, measures)

    # this is temporary
    @staticmethod
    def get_all(session: Session) -> List[Row]:
        return session.execute(select(Measure.sensorId, Measure.timestamp, Measure.data)).all()








    ######################## BELOW MUST BE RE-IMPLEMENTED #########################
    @staticmethod
    def upload_measures(session: Session, files):
        if len(files.keys()) == 0:
            raise FileNotFoundError()
        key = RSA.import_key(open('./resources/public.pem').read())
        outcomes = []
        for f in files.keys():
            if ".dat" in f:
                out = {"filename": f, "status": MeasureStatus.VALID.name}
                if files.get(f).filename != f:
                    out["status"] = MeasureStatus.FILENAME_INCONSISTENCY.name
                fileName = f.split(".")[0]
                signFileName = fileName + ".sig"
                if signFileName in files.keys():
                    try:
                        verify_signature(key, files.get(f), files.get(signFileName))
                    except (ValueError, TypeError):
                        out["status"] = MeasureStatus.SIGN_NOT_VALID.name
                    else:
                        measures = bytes_to_measures(files.get(f))
                        session.add_all(measures)
                        session.commit()
                else:
                    out["status"] = MeasureStatus.NO_SIGNATURE.name
                outcomes.append(out)
        return outcomes


    @staticmethod
    def get_first_result_by_sensor_id(session: Session, sensorId: int):
        return session.query(Measure)\
            .filter(Measure.sensorId == sensorId)\
            .order_by(Measure.timestamp.asc())\
            .first()


    @staticmethod
    def get_measures_by_pm_and_board_id(session: Session, pm: str, bs_id, params):
        if not pm == "pm10" and not pm == "pm25":
            return None

        start = params['start']
        end = params['end']

        if not start or not end:
            return None

        results = session.query(
                FiveMinAvgMeasure,
                LogicalSensor
            ).with_entities(
                FiveMinAvgMeasure.date,
                FiveMinAvgMeasure.hour,
                FiveMinAvgMeasure.minute,
                func.group_concat(LogicalSensor.description + ':' + expression.cast(FiveMinAvgMeasure.sensorId, types.CHAR) + ':' + expression.cast(FiveMinAvgMeasure.value, types.CHAR), ',').label('data')
            ).filter(LogicalSensor.sensorId == FiveMinAvgMeasure.sensorId) \
            .join(PhysicalSensor, LogicalSensor.phSensorId == PhysicalSensor.sensorId) \
            .join(BoardConnection, PhysicalSensor.connectionId == BoardConnection.connectionId) \
            .filter(BoardConnection.boardId == bs_id) \
            .filter(PhysicalSensor.description.in_(('PM', 'DHT'))) \
            .filter(LogicalSensor.description.in_((pm, 'DHT_T', 'DHT_H'))) \
            .filter(FiveMinAvgMeasure.date >= start) \
            .filter(FiveMinAvgMeasure.date <= end) \
            .group_by(FiveMinAvgMeasure.date, FiveMinAvgMeasure.hour, FiveMinAvgMeasure.minute)

        response = []
        for r in results:
            measures_on_time = {
                "date": str(r.date),
                "hour": r.hour,
                "minute": r.minute
            }

            for d in r.data.split(','):
                info = d.split(':')
                sensor_name = 's' + info[1] + ('_cal' if info[0] == pm else '')
                measures_on_time[sensor_name] = float(info[2])

            response.append(measures_on_time)

        return response


    @staticmethod
    def get_measures_by_pm(session: Session, pm: str):
        if not pm == "pm10" and not pm == "pm25":
            return None

        # retrieve correct BoardExperiment by startTime and endTime of he measure to find the board location
        boardExperimentId = session.query(
                BoardExperiment.boardExperimentId
            ).filter(BoardExperiment.boardId == Board.boardId) \
            .filter(BoardExperiment.startTime <= func.unixepoch(FiveMinAvgMeasure.date) + FiveMinAvgMeasure.hour * 3600 + FiveMinAvgMeasure.minute * 60) \
            .filter(or_(
                BoardExperiment.endTime == None,
                BoardExperiment.endTime >= func.unixepoch(FiveMinAvgMeasure.date) + FiveMinAvgMeasure.hour * 3600 + FiveMinAvgMeasure.minute * 60
            )).order_by(BoardExperiment.startTime.desc(), BoardExperiment.endTime.desc()) \
            .limit(1) \
            .correlate(FiveMinAvgMeasure, Board) \
            .as_scalar()

        # retrieve last FiveMinAvgMeasure by sensorId
        measureId = session.query(
                func.max(FiveMinAvgMeasure.id).label('id')
            ).filter(FiveMinAvgMeasure.sensorId == LogicalSensor.sensorId) \
            .correlate(LogicalSensor) \
            .as_scalar()

        # retrieve last PM FiveMinAvgMeasure for every PM sensor of all the boards
        pmMeasures = session.query(
                Board.boardId,
                BoardConnection.connectionId,
                FiveMinAvgMeasure,
                BoardExperiment.latitude,
                BoardExperiment.longitude,
            ).filter(Board.boardId == BoardConnection.boardId) \
            .filter(PhysicalSensor.connectionId == BoardConnection.connectionId) \
            .filter(LogicalSensor.phSensorId == PhysicalSensor.sensorId) \
            .filter(BoardExperiment.boardExperimentId == boardExperimentId) \
            .filter(BoardExperiment.boardId == Board.boardId) \
            .filter(PhysicalSensor.description == "PM") \
            .filter(LogicalSensor.description == pm) \
            .filter(FiveMinAvgMeasure.id == measureId) \
            .filter(FiveMinAvgMeasure.sensorId == LogicalSensor.sensorId) \
            .subquery()

        results = pmMeasures
        for dht in ["DHT_T", "DHT_H"]:
            dhtMeasure = session.query(
                    FiveMinAvgMeasure.date,
                    FiveMinAvgMeasure.hour,
                    FiveMinAvgMeasure.minute,
                    FiveMinAvgMeasure.value,
                    BoardConnection.connectionId
                ).filter(Board.boardId == BoardConnection.boardId) \
                .filter(PhysicalSensor.connectionId == BoardConnection.connectionId) \
                .filter(LogicalSensor.phSensorId == PhysicalSensor.sensorId) \
                .filter(PhysicalSensor.description == "DHT") \
                .filter(LogicalSensor.description == dht) \
                .filter(FiveMinAvgMeasure.sensorId == LogicalSensor.sensorId) \
                .subquery()

            # add to every record DHT measure (temperature and humidity)
            results = session.query(
                    results
                ).with_entities(results, dhtMeasure.c.value.label(dht)) \
                .join(dhtMeasure,
                    and_(results.c.date == dhtMeasure.c.date,
                    and_(results.c.hour == dhtMeasure.c.hour,
                    and_(results.c.minute == dhtMeasure.c.minute,
                    results.c.connectionId == dhtMeasure.c.connectionId))),
                    isouter=True
                ).subquery()

        results = session.query(results)

        response = {'boards': []}
        for r in results:
            print(r)
            try:
                boardIndex = list(map(lambda b: b['id'], response['boards'])).index(r.boardId)
            except ValueError:
                response['boards'].append({'id': r.boardId, 'sensors': []})
                boardIndex = -1

            sensor = {}
            sensor['id'] = str(r.sensorId)
            sensor['date'] = str(r.date)
            sensor['hour'] = str(r.hour)
            sensor['minute'] = str(r.minute)
            sensor['value'] = str(r.value)
            sensor['max'] = str(r.max)
            sensor['min'] = str(r.min)
            sensor['std'] = str(r.std)

            response['boards'][boardIndex]['sensors'].append(sensor)
            response['boards'][boardIndex]['temp'] = str(r.DHT_T) if r.DHT_T else None
            response['boards'][boardIndex]['rh'] = str(r.DHT_H) if r.DHT_H else None
            response['boards'][boardIndex]['lat'] = str(r.latitude)
            response['boards'][boardIndex]['long'] = str(r.longitude)

        return response