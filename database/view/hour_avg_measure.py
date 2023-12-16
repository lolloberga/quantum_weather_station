from database.model.base.database import Base
from sqlalchemy import Column, BigInteger, Integer, Date, Float


class HourAvgMeasure(Base):

    __tablename__ = "hour_avg_measure"

    id = Column(BigInteger().with_variant(
        Integer, "sqlite"), primary_key=True)

    timestamp = Column(BigInteger(), nullable = False)
    sensorId = Column(BigInteger(), nullable=False)

    data = Column(Float(), nullable=False)

