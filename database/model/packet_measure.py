from database.model.base.database import Base
from sqlalchemy import Column, BigInteger, Integer, ForeignKey, Float, Index, DateTime, UniqueConstraint
from sqlalchemy.dialects.mysql import DATETIME, SMALLINT, DOUBLE


class PacketMeasure(Base):

    __tablename__ = "packet_measure"

    measureId = Column(
        BigInteger().with_variant(Integer, "sqlite"),
        primary_key=True)

    sensorId = Column(
        Integer().with_variant(SMALLINT(unsigned=True), "mysql"),
        ForeignKey('logical_sensor.sensorId'), nullable=False
    )

    timestamp = Column(DateTime().with_variant(DATETIME(0), "mysql"))
    
    data = Column(Float().with_variant(DOUBLE, "mysql"), nullable=False)

    packetId = Column(
        BigInteger().with_variant(Integer, "sqlite"),
        ForeignKey('packet.packetId'), nullable=False,
    )

    position = Column(
        Integer().with_variant(SMALLINT(unsigned=True), "mysql")
    )

UniqueConstraint(PacketMeasure.packetId, PacketMeasure.position)