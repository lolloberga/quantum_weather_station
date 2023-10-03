from typing import List, Tuple
from sqlalchemy.orm import Session
from db_lib.models import Board, LogicalSensor, UnitOfMeasure
from db_lib.services.board_experiments_service import BoardExperimentsService
import geopy.distance
import time

from sqlalchemy import select


class BoardService:
    @staticmethod
    def get_all(session: Session, modelId=None) -> List[Board]:
      if modelId is None:
        return session.query(Board).all()
      else:
        return session.query(Board).filter(Board.vendorModelId == modelId)

    @staticmethod
    def get_all_ids(session: Session, modelId=None) -> List[int]:
        if modelId is None:
            return [r.boardId for r in session.query(Board.boardId).all()]
        else:
            return [r.boardId for r in session.query(Board.boardId).filter(Board.vendorModelId == modelId).all()]

    @staticmethod
    def get_by_id(session: Session, boardId: int) -> Board:
        return session.get(Board, boardId)
    
    @staticmethod
    def get_sensors(session: Session, boardId: int) -> List[Tuple[LogicalSensor, UnitOfMeasure]]:
        query = select(LogicalSensor, UnitOfMeasure).join_from(
            LogicalSensor, UnitOfMeasure, LogicalSensor.unitId == UnitOfMeasure.unitId, isouter=True
        ).filter(LogicalSensor.boardId==boardId)
        return session.execute(query).all()

    @staticmethod
    def get_all_by_location(session: Session, latitude: float, longitude: float, maxDistanceKm):
        boardIds = BoardService.get_all_ids(session)
        boardExperiments = BoardExperimentsService.get_board_experiments_between(boardIds, int(time.time()))
        nearbyBoards = []
        for exp in boardExperiments:
            distanceKm = geopy.distance.geodesic((latitude, longitude), (exp.latitude, exp.longitude)).km
            print("distanceKm: " + str(distanceKm))
            if distanceKm <= maxDistanceKm:
                nearbyBoards.append(exp)
        return nearbyBoards

    @staticmethod
    def create_board(session: Session, board: Board) -> Board:
        session.add(board)
        session.flush()
        return board

    @staticmethod
    def delete_board(session: Session, boardId: int) -> None:
        session.delete(session.query(Board).get(boardId))
        session.flush()

    @staticmethod
    def add_sensor_to_board(session: Session, boardId: int, sensorId: int) -> None:
        board = session.query(Board).get(boardId)
        sensor = session.query(PhysicalSensor).get(sensorId)
        connection = BoardConnection(
            timestamp="AAA",
            sensor=sensor,
            boardId=board.boardId,
            boardPin=10
        )

        board.connections.append(connection)
        session.flush()
