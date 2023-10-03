from typing import List
from dataset.model.param_type import ParamType
from sqlalchemy.orm import Session


class ParamTypeService:
    @staticmethod
    def get_all(session: Session) -> List[ParamType]:
        return session.query(ParamType).all()