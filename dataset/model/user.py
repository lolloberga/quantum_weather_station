from db_lib.database import Base
from sqlalchemy.orm import relationship
from sqlalchemy import Column, BigInteger, Integer, Boolean, String, Date


class User(Base):

    __tablename__ = "user"

    user_id = Column(BigInteger().with_variant(Integer, "sqlite"), primary_key=True)
    email = Column(String(120), nullable=False, unique=True)
    password = Column(String(60), nullable=False)
    name = Column(String(30), nullable=False)
    surname = Column(String(30), nullable=False)
    birth = Column(Date, nullable=False)
    role = Column(String(30), nullable=False)
    confirmed = Column(Boolean, nullable=False)
    active = Column(Boolean, nullable=False)
    counter = Column(BigInteger, nullable=False)
    reset_pass = Column(Boolean, nullable=False)