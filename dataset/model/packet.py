from db_lib.database import Base
from sqlalchemy import Column, BigInteger, Integer, ForeignKey, UniqueConstraint, DateTime
from sqlalchemy.dialects.mysql import DATETIME


class Packet(Base):

    __tablename__ = "packet"

    # with variant is needed if you want to auto-generate keys
    packetId = Column(BigInteger().with_variant(
        Integer, "sqlite"), primary_key=True)

    boardId = Column(Integer(), ForeignKey('board.boardId'), nullable=False)

    timestamp = Column(DateTime().with_variant(DATETIME(0), "mysql"), nullable=False)

UniqueConstraint(Packet.boardId, Packet.timestamp)