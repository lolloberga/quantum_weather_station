from db_lib.database import Base
from sqlalchemy.orm import relationship
from sqlalchemy import Column, Integer, String


class UnitOfMeasure(Base):

    __tablename__ = "unit_of_measure"

    unitId = Column(Integer(), primary_key=True)

    measureName = Column(String(30), nullable=False)
    unitOfMeasure = Column(String(30), nullable=False)
