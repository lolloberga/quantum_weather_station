from typing import List
from dataset.model.unit_of_measure import UnitOfMeasure
from sqlalchemy.orm import Session


class UnitOfMeasureService:
    @staticmethod
    def get_all(session: Session) -> List[UnitOfMeasure]:
        return session.query(UnitOfMeasure).all()

    @staticmethod
    def get_by_id(session: Session, unitId: int) -> UnitOfMeasure:
        return session.query(UnitOfMeasure).get(unitId)

    @staticmethod
    def create(session: Session, unit: UnitOfMeasure) -> UnitOfMeasure:
        session.add(unit)
        session.flush()
        return unit

    @staticmethod
    def delete(session: Session, unitId: int) -> None:
        session.delete(session.query(UnitOfMeasure).get(unitId))
        session.flush()

