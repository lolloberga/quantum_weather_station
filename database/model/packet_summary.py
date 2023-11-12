from dataset.model.base.database import Base
from sqlalchemy import Column, BigInteger, Integer, ForeignKey, String, UniqueConstraint, DateTime
from sqlalchemy.dialects.mysql import DATETIME


class PacketSummary(Base):

    __tablename__ = "packet_summary"

    # with variant is needed if you want to auto-generate keys
    packetId = Column(BigInteger().with_variant(
        Integer, "sqlite"), primary_key=True)

    boardId = Column(Integer(), ForeignKey('board.boardId'), nullable=False)

    created = Column(DateTime().with_variant(DATETIME(0), "mysql"), nullable=False)

    ip = Column(String(45), nullable=False)

    sha256 = Column(String(64), nullable=False)

UniqueConstraint(PacketSummary.boardId, PacketSummary.created)