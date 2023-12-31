from sqlalchemy.orm import Session
from database.model.packet import Packet
from database.model.measure import Measure
from database.service.packet_measure_service import PacketMeasureService
from database.service.measure_service import MeasureService
from sqlalchemy import select
from typing import List, Dict, Any


class PacketService:
    @staticmethod
    def add_packet(session: Session, packet: Packet, measures: List[Dict[str, Any]] = None) -> Packet:
        session.add(packet)
        session.flush()
        PacketMeasureService.add_measurements_bulk(session, packet.packetId, measures)
        MeasureService.add_measurements_bulk(session, measures, ignore=True)
        return Packet
        

    @staticmethod
    def get_by_id(session: Session, packet_id: int) -> Packet:
        return session.get(Packet, packet_id)
        
