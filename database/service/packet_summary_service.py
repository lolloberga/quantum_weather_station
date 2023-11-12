from sqlalchemy.orm import Session
from dataset.model.packet_summary import PacketSummary
from dataset.model.packet_connection import PacketConnection
from dataset.model.packet import Packet
from sqlalchemy import select, and_, func, join
from typing import List
from datetime import datetime


class PacketSummaryService:
    @staticmethod
    def create(session: Session, packet_summary: PacketSummary, timestamps: List[datetime]) -> None:
        # add summary packet
        session.add(packet_summary)
        session.flush()

        # save timestamps in packet connection
        for idx, ts in enumerate(timestamps):
            session.add(PacketConnection(packetSummaryId=packet_summary.packetId, timestamp=ts, packetPosition=idx))
        session.flush()
        

    @staticmethod
    def get_by_id(session: Session, packet_id: int) -> PacketSummary:
        packet_summary = session.get(PacketSummary, packet_id)
        return packet_summary
        

    @staticmethod
    def get_timestamps(session: Session, packet_id: int) -> List[datetime]:
        timestamps = session.execute(
            select(PacketConnection.timestamp)
            .where(PacketConnection.packetSummaryId==packet_id)
            .order_by(PacketConnection.packetPosition.asc())
        ).all()
        return [ts[0] for ts in timestamps]


    @staticmethod
    def get_in_interval(session: Session, start_ts: datetime, end_ts: datetime, board_id = None) -> List[PacketSummary]:
        query = None
        if board_id is not None:            
            query = select(PacketSummary).where(
                and_(
                    PacketSummary.created >= start_ts,
                    PacketSummary.created <= end_ts,
                    PacketSummary.boardId == board_id
                )
            )
        else:
            query = select(PacketSummary).where(
                and_(
                    PacketSummary.created >= start_ts,
                    PacketSummary.created <= end_ts
                )
            )
        return [row[0] for row in session.execute(query).all()]

    @staticmethod
    def get_contained_packets(session: Session, packet_id: int) -> List[Packet]:
        packets = session.execute(
            select(Packet)
            .join_from(PacketConnection, PacketSummary)
            .join(Packet, and_(Packet.timestamp == PacketConnection.timestamp, Packet.boardId == PacketSummary.boardId))
            .where(PacketSummary.packetId == packet_id)
            .order_by(PacketConnection.packetPosition.asc())
        ).all()
        return [packet[0] for packet in packets]
        
    @staticmethod
    def get_contained_packet_ids(session: Session, packet_id: int) -> List[int]:
        packet_ids = session.execute(
            select(Packet.packetId)
            .join_from(PacketConnection, PacketSummary)
            .join(Packet, and_(Packet.timestamp == PacketConnection.timestamp, Packet.boardId == PacketSummary.boardId))
            .where(PacketSummary.packetId == packet_id)
            .order_by(PacketConnection.packetPosition.asc())
            .distinct()
        ).all()
        return [pk_id[0] for pk_id in packet_ids]        