from dataset.model.base.database import Base
from sqlalchemy import Column, Integer, ForeignKey, Float, Index, DateTime
from sqlalchemy.dialects.mysql import DATETIME, SMALLINT, DOUBLE


class MeasureTemporary(Base):

    __tablename__ = "measure_temp"

    id = Column(Integer, primary_key=True, autoincrement=True)
    
    sensorId = Column(
        Integer().with_variant(SMALLINT(unsigned=True), "mysql"),
        ForeignKey('logical_sensor.sensorId'), nullable=False,
        primary_key=False
    )

    timestamp = Column(
        DateTime().with_variant(DATETIME(0), "mysql"),
        primary_key=False
    )
    
    data = Column(Float().with_variant(DOUBLE, "mysql"), nullable=False)
    