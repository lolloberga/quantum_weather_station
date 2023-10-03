from typing import List
from db_lib.models.vendor_model import VendorModel
from sqlalchemy.orm import Session


class ModelsService:
    @staticmethod
    def get_all(session: Session) -> List[VendorModel]:
        return session.query(VendorModel).all()

    @staticmethod
    def get_by_id(session: Session, modelId: int) -> VendorModel:
        return session.query(VendorModel).get(modelId)

    @staticmethod
    def create(session: Session, model: VendorModel) -> VendorModel:
        session.add(model)
        session.commit()
        return model

    @staticmethod
    def delete(session: Session, modelId: int) -> None:
        session.delete(session.query(VendorModel).get(modelId))
        session.commit()

