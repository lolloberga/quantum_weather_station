from typing import List
from sqlalchemy.orm import Session
from db_lib.models.board_config import BoardConfig
from sqlalchemy import func


class BoardConfigService:
    @staticmethod
    def get_all(session: Session, boardIds: List[int] = None, paramIds: List[int] = None) -> List[BoardConfig]:
        # Query to extract latest configuration from duplicates (indicated by paramId) by boardId
        rankQuery = session.query(
            BoardConfig,
            func.rank().over(
                order_by=BoardConfig.timestamp.desc(),
                partition_by=(BoardConfig.boardId, BoardConfig.paramId)
            ).label('rnk')
        )

        if boardIds:
            rankQuery = rankQuery.filter(BoardConfig.boardId.in_(boardIds))
        if paramIds:
            rankQuery = rankQuery.filter(BoardConfig.paramId.in_(paramIds))

        rankQuery = rankQuery.subquery()
        return session.query(rankQuery).filter(rankQuery.c.rnk == 1)