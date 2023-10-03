from db_lib.database import Base
from sqlalchemy import Column, Integer, ForeignKey, String, Time
from db_lib.models.unit_of_measure import UnitOfMeasure
from db_lib.models.measure import Measure
from sqlalchemy.dialects.mysql import SMALLINT


class LogicalSensor(Base):

    __tablename__ = "logical_sensor"

    sensorId = Column(
        Integer().with_variant(SMALLINT(unsigned=True), "mysql"),
         primary_key=True,  autoincrement=False)

    boardId = Column(
        Integer(), ForeignKey('board.boardId'), nullable=False)

    unitId = Column(
        Integer(), ForeignKey('unit_of_measure.unitId'), nullable=False)

    acqTime = Column(Time())
    description = Column(String(256))

    

    
