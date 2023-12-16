from database.model.base.database import Base
from sqlalchemy import Column, Integer, ForeignKey, Float, Index, DateTime
from sqlalchemy.dialects.mysql import DATETIME, SMALLINT, DOUBLE


class Measure(Base):

    __tablename__ = "measure"

    sensorId = Column(
        Integer().with_variant(SMALLINT(unsigned=True), "mysql"),
        ForeignKey('logical_sensor.sensorId'), nullable=False,
        primary_key = True
    )

    timestamp = Column(
        DateTime().with_variant(DATETIME(0), "mysql"),
        primary_key=True
    )
    
    data = Column(Float().with_variant(DOUBLE, "mysql"), nullable=False)
    