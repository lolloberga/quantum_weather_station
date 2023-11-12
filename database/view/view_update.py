from dataset.model.base.database import Base
from sqlalchemy import Column, BigInteger, String


class ViewUpdate(Base):

    __tablename__ = "view_update"

    name = Column(String(60), primary_key=True)
    lastUpdate = Column(BigInteger(), nullable=False)
