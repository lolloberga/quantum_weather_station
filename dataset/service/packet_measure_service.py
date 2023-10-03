from sqlalchemy.orm import Session
from db_lib.models import Packet, PacketMeasure, PacketConnection, PacketSummary
from db_lib.services.packet_summary_service import PacketSummaryService
from sqlalchemy.sql.selectable import Select
from sqlalchemy import insert, select, and_
from typing import List, Dict, Any
from sqlalchemy.engine import Row


class PacketMeasureService:
    @staticmethod
    def add_measurements_bulk(session: Session, packetId: int, measures: List[Dict[str, Any]]) -> None:
        # return if there are no measurements
        if not measures:
            return
        # this is a fast bulk insert (only one query, with no object creation)
        packet_measures = [dict(measure, packetId=packetId , position=pos) for pos, measure in enumerate(measures, start=1)]
        session.execute(insert(PacketMeasure), packet_measures)


    @staticmethod
    def get_from_packet_summary(session: Session, packet_summary_id: int) -> List[Row]:
        # subquery gets packets and their order
        subquery = (select(Packet.packetId, PacketConnection.packetPosition)
            .join_from(PacketConnection, PacketSummary)
            .join(Packet, and_(Packet.timestamp == PacketConnection.timestamp, Packet.boardId == PacketSummary.boardId))
            .where(PacketSummary.packetId == packet_summary_id)
            .order_by(PacketConnection.packetPosition.asc()).distinct().subquery())
        # query packet measures and order by packer oder and measure order
        query = select(
            PacketMeasure.packetId,
            PacketMeasure.sensorId,
            PacketMeasure.timestamp,
            PacketMeasure.data
        ).join(subquery, PacketMeasure.packetId==subquery.c.packetId).order_by(subquery.c.packetPosition.asc(), PacketMeasure.position.asc())
        return session.execute(query).all()
    

    @staticmethod
    def get_from_packet(session: Session, packetId: int) -> List[Row]:
        query = select(
            PacketMeasure.sensorId,
            PacketMeasure.timestamp,
            PacketMeasure.data
        ).where(PacketMeasure.packetId==packetId).order_by(PacketMeasure.position.asc())
        return session.execute(query).all()