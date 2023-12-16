from database.model.base.database import Base
from sqlalchemy.orm import relationship
from sqlalchemy import Column, BigInteger, Integer, Float, String
from database.model.board_experiment import BoardExperiment


class Experiment(Base):

    __tablename__ = "experiment"

    experimentId = Column(Integer(), primary_key=True)

    name = Column(String(60), nullable=False)
    
    description = Column(String(256))