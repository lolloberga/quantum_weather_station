from db_lib.database import Base
from sqlalchemy import Column, BigInteger, Integer, ForeignKey, DateTime
from sqlalchemy.dialects.mysql import DATETIME


class PacketConnection(Base):

    __tablename__ = "packet_connection"

    # with variant is needed if you want to auto-generate keys
    packetConnectionId = Column(BigInteger().with_variant(
        Integer, "sqlite"), primary_key=True) # this also gives the order the packets in the summary

    packetSummaryId = Column(BigInteger()
        .with_variant(Integer, "sqlite"), ForeignKey('packet_summary.packetId'), nullable=False)

    timestamp = Column(
        DateTime().with_variant(DATETIME(0), "mysql"),
        nullable=False
    )

    packetPosition = Column(Integer(), nullable=False)