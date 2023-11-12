from typing import List

from sqlalchemy import func
from sqlalchemy.orm import Session

from dataset.model.board import Board
from dataset.model.experiment import Experiment
from dataset.model.board_experiment import BoardExperiment
from operator import and_, or_


class BoardExperimentsService:
    @staticmethod
    def get_all(session: Session) -> List[BoardExperiment]:
        return session.query(BoardExperiment).all()

    @staticmethod
    def get_by_id(session: Session, boardExperimentId: int) -> BoardExperiment:
        return session.query(BoardExperiment).get(boardExperimentId)

    @staticmethod
    def connect(session: Session, experimentId: int, boardId: int) -> BoardExperiment:
        experiment = session.query(Experiment).get(experimentId)
        board = session.query(Board).get(boardId)
        boardExperiment = BoardExperiment(
            board=board,
            experiment=experiment
            # TODO: Add othe fields
        )
        
        session.add(boardExperiment)
        session.commit()
        return boardExperiment

    @staticmethod
    def get_board_experiments_between(session: Session, boardIds: List[int], timestamp):
        rankQuery = session.query(
                BoardExperiment,
                func.rank().over(
                    order_by=(BoardExperiment.startTime.desc(), BoardExperiment.endTime.desc()),
                    partition_by=BoardExperiment.boardId
                ).label('rnk')
            ).filter(BoardExperiment.boardId.in_(boardIds))\
            .filter(BoardExperiment.startTime <= timestamp)\
            .filter(or_(BoardExperiment.endTime == None, BoardExperiment.endTime >= timestamp))\
            .subquery()
        return session.query(rankQuery).with_entities(BoardExperiment).filter(rankQuery.c.rnk == 1).all()