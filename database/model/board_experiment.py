from dataset.model.base.database import Base
from sqlalchemy import Column, DateTime, Integer, ForeignKey, Float
from sqlalchemy.dialects.mysql import DATETIME


class BoardExperiment(Base):

    __tablename__ = "board_experiment"

    boardExperimentId = Column(Integer(), primary_key=True)

    experimentId = Column(
        Integer(), ForeignKey('experiment.experimentId'), nullable=False)
    
    boardId = Column(Integer(), ForeignKey('board.boardId'), nullable=False)

    startTime = Column(DateTime().with_variant(DATETIME(0), "mysql"))
    endTime = Column(DateTime().with_variant(DATETIME(0), "mysql"))    

    latitude = Column(Float())
    longitude = Column(Float())
    altitude = Column(Float())
