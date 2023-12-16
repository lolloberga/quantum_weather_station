from database.model.base.database import Base
from sqlalchemy.orm import relationship
from sqlalchemy import Column, BigInteger, Integer, String
from database.model.physical_sensor import PhysicalSensor


class VendorModel(Base):

    __tablename__ = "vendor_model"

    modelId = Column(Integer(), primary_key=True)

    vendor = Column(String(60), nullable=False)
    model = Column(String(60), nullable=False)
