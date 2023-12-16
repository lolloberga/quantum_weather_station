from database.model.base.database import Base
from sqlalchemy import Column, BigInteger, Integer, Float, Date


class FiveMinAvgMeasure(Base):

    __tablename__ = "five_min_avg_measure"

    id = Column(BigInteger().with_variant(
        Integer, "sqlite"), primary_key=True)

    timestamp = Column(BigInteger(), nullable = False)
    sensorId = Column(BigInteger(), nullable=False)

    value = Column(Float(), nullable=False)
    max = Column(Float(), nullable=True)
    min = Column(Float(), nullable=True)
    std = Column(Float(), nullable=True)

    # latitude = Column(Float())
    # longitude = Column(Float())

