from typing import List
from sqlalchemy.orm import Session
from database.model.board import Board
from database.model.logical_physical_connection import LogicalPhysicalConnection
from database.model.physical_sensor import PhysicalSensor


class LogicalPhysicalConnectionService:
    pass
    
    # @staticmethod
    # def get_all(session: Session, boardId = None, sensorId = None) -> List[BoardConnection]:
    #   return session.query(BoardConnection).filter(BoardConnection.boardId == boardId)

    # @staticmethod
    # def get_by_id(session: Session, connectionId: int) -> BoardConnection:
    #     return session.query(BoardConnection).get(connectionId)

    # @staticmethod
    # def update_board_pin(session: Session, connectionId: int, value: int) -> BoardConnection:
    #     connection = session.query(BoardConnection).get(connectionId)
    #     connection.boardPin = value
    #     session.commit()
    #     return connection

    # @staticmethod
    # def connect(session: Session, boardId: int, sensorId: int) -> BoardConnection:
    #     board = session.query(Board).get(boardId)
    #     sensor = session.query(PhysicalSensor).get(sensorId)
    #     connection = BoardConnection(
    #         timestamp="AAA",
    #         sensor=sensor,
    #         boardId=board.boardId,
    #         boardPin=10
    #     )

    #     session.add(connection)
    #     session.commit()
    #     return connection

    # @staticmethod
    # def disconnect(session: Session, boardId: int, sensorId: int) -> BoardConnection:
    #     sensor = session.query(PhysicalSensor).get(sensorId)
    #     connection = session.query(BoardConnection).filter(BoardConnection.connectionId == sensor.connectionId).first()

    #     session.delete(connection)
    #     session.commit()
    #     return connection