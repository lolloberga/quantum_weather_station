from db_lib.database import Base
from sqlalchemy.orm import relationship
from sqlalchemy import Column, Integer, ForeignKey, String, UniqueConstraint
from db_lib.models.board_experiment import BoardExperiment
from db_lib.models.board_config import BoardConfig


class Board(Base):

    __tablename__ = "board"

    boardId = Column(Integer(), primary_key=True, autoincrement=False)

    vendorModelId = Column(
        Integer(), ForeignKey('vendor_model.modelId'))

    serialNumber = Column(String(60))

UniqueConstraint(
    Board.serialNumber
)