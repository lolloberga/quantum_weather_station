from typing import List
from database.model.logical_sensor import LogicalSensor
from database.model.physical_sensor import PhysicalSensor
from sqlalchemy.orm import Session


class PhysicalSensorService:
    @staticmethod
    def get_all(session: Session, modelId: int = None) -> List[PhysicalSensor]:
        if modelId is None:
          return session.query(PhysicalSensor).all()
        else:
          return session.query(PhysicalSensor).filter(PhysicalSensor.vendorModelId == modelId)

    @staticmethod
    def get_by_id(session: Session, sensorId: int) -> PhysicalSensor:
        return session.query(PhysicalSensor).get(sensorId)

    @staticmethod
    def create(session: Session, modelId: int, sensor: PhysicalSensor) -> PhysicalSensor:
        sensor.vendorModelId = modelId
        session.add(sensor)
        session.commit()
        return sensor

    @staticmethod
    def delete(session: Session, modelId:int, sensorId: int) -> None:
        sensor = session.query(PhysicalSensor).get(sensorId)
        connection = session.query(BoardConnection).get(sensor.connectionId)
        session.delete(sensor)
        session.delete(connection)
        session.commit()

    @staticmethod
    def add_logical_sensor(session: Session, sensorId: int, logical_sensor: LogicalSensor) -> None:
        sensor = session.query(PhysicalSensor).get(sensorId)
        logical_sensor.physicalSensor = sensor
        sensor.logicalSensors.append(logical_sensor)
        session.commit()
