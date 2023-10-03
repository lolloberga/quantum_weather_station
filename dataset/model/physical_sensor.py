from dataset.model.base.database import Base
from sqlalchemy.orm import relationship
from sqlalchemy import Column, DateTime, Integer, ForeignKey, String
from dataset.model.logical_sensor import LogicalSensor
from sqlalchemy.dialects.mysql import DATETIME


class PhysicalSensor(Base):

    __tablename__ = "physical_sensor"

    sensorId = Column(Integer(), primary_key=True, autoincrement=False)
    
    vendorModelId = Column(
        Integer(), ForeignKey('vendor_model.modelId'))

    serialNumber = Column(String(60))
    firstUse = Column(DateTime().with_variant(DATETIME(0), "mysql"))
    description = Column(String(256))

    
