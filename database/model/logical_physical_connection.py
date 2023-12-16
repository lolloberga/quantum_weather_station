from database.model.base.database import Base
from sqlalchemy.orm import relationship
from sqlalchemy import Column, DateTime, Integer, ForeignKey, UniqueConstraint
from database.model.physical_sensor import PhysicalSensor
from sqlalchemy.dialects.mysql import SMALLINT, DATETIME


class LogicalPhysicalConnection(Base):

    __tablename__ = "logical_physical_connection"

    connectionId = Column(Integer(), primary_key=True)

    logicSensorId = Column(
            Integer().with_variant(SMALLINT(unsigned=True), "mysql"),
            ForeignKey('logical_sensor.sensorId'),
            nullable=False
    )

    phSensorId = Column(Integer(), ForeignKey('physical_sensor.sensorId'), nullable=False)

    timestamp = Column(DateTime().with_variant(DATETIME(0), "mysql"), nullable=False)
    
    boardPin = Column(Integer(),  nullable=False)


UniqueConstraint(
    LogicalPhysicalConnection.logicSensorId,
    LogicalPhysicalConnection.timestamp
)

   
