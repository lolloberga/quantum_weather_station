from dataset.model.base.database import Base
from sqlalchemy import Column, BigInteger, Integer, String


class ParamType(Base):

    __tablename__ = "param_type"

    paramId = Column(Integer(), primary_key=True)

    name = Column(String(30), nullable=False)

    description = Column(String(256))

