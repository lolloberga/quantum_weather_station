from database.model.base.database import Base
from database.model.param_type import ParamType
from sqlalchemy import Column, DateTime, Integer, ForeignKey, String, UniqueConstraint
from sqlalchemy.dialects.mysql import DATETIME


class BoardConfig(Base):

    __tablename__ = "board_config"

    configId = Column(Integer(), primary_key=True)

    boardId = Column(
        Integer(), ForeignKey('board.boardId'), nullable=False)

    paramId = Column(
        Integer(), ForeignKey('param_type.paramId'), nullable=False)

    timestamp = Column(DateTime().with_variant(DATETIME(0), "mysql"), nullable=False)

    paramValue = Column(String(60), nullable=False)

UniqueConstraint(BoardConfig.boardId, BoardConfig.paramId, BoardConfig.timestamp)
