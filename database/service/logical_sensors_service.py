from typing import List
from sqlalchemy import select, and_, func
from sqlalchemy.orm import Session
from database.model.logical_sensor import LogicalSensor
from database.model.physical_sensor import PhysicalSensor
from database.model.board import Board


class LogicalSensorService:
    @staticmethod
    def get_all(session: Session, phSensorId: int) -> List[LogicalSensor]:
        if int(phSensorId) == -1:
            return session.query(LogicalSensor).all()
        else:
            return session.query(LogicalSensor).filter(LogicalSensor.phSensorId == phSensorId).all()


    @staticmethod
    def get_by_id(session: Session, phSensorId: int, sensorId: int) -> LogicalSensor:
        return session.query(LogicalSensor).filter(LogicalSensor.phSensorId == phSensorId & LogicalSensor.sensorId == sensorId).first()


    @staticmethod
    def create(session: Session, sensor: LogicalSensor) -> LogicalSensor:
        session.add(sensor)
        session.flush()
        return sensor


    @staticmethod
    def delete(session: Session, phSensorId: int, sensorId: int) -> None:
        sensor = session.query(LogicalSensor).filter(
            LogicalSensor.phSensorId == phSensorId & LogicalSensor.sensorId == sensorId).first()
        session.delete(sensor)
        session.flush()

    
    @staticmethod
    def get_all_by_board(session: Session, boardId: int) -> List[LogicalSensor]:
        rows = session.execute(
            select(LogicalSensor)
            .where(LogicalSensor.boardId==boardId)
            .order_by(LogicalSensor.sensorId)
        )
        return [r[0] for r in rows]
        
        
        
    
