from db_lib.database import Base
from sqlalchemy.orm import relationship
from sqlalchemy import Column, BigInteger, Integer, String
from db_lib.models.board import Board
from db_lib.models.physical_sensor import PhysicalSensor


class VendorModel(Base):

    __tablename__ = "vendor_model"

    modelId = Column(Integer(), primary_key=True)

    vendor = Column(String(60), nullable=False)
    model = Column(String(60), nullable=False)
