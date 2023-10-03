import os

__flask__ = os.getenv("SQL_FLASK", "disable")

Base = None
if __flask__ == "enable":
    from flask_sqlalchemy import SQLAlchemy
    db = SQLAlchemy()
    Base = db.Model

else:
    from sqlalchemy.orm import declarative_base    
    Base = declarative_base()





