from database.model.base.database import Base
from database.model.vendor_model import VendorModel
from sqlalchemy.orm import relationship
from sqlalchemy import Column, Integer, ForeignKey, String, UniqueConstraint
from database.model.board_experiment import BoardExperiment
from database.model.board_config import BoardConfig


class Board(Base):

    __tablename__ = "board"

    boardId = Column(Integer(), primary_key=True, autoincrement=False)

    vendorModelId = Column(
        Integer(), ForeignKey('vendor_model.modelId'))

    serialNumber = Column(String(60))

UniqueConstraint(
    Board.serialNumber
)