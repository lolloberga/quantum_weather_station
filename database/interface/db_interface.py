from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy_utils import database_exists, create_database

# this import adds metadata on ORM mappping
from dataset.model.base.database import Base

# usefull to pupulate metadata object
import dataset.model
import dataset.view


class DBUrl():
    def __init__(self):
        self._url=None
    
    def get_url(self):
        return self._url

class SQLiteUrl(DBUrl):
    def __init__(self, db_path: str):
        super().__init__()
        self._url = "sqlite:///{}".format(db_path)


class MySQLUrl(DBUrl):
    def __init__(self, db_name: str, user: str, password: str, host: str, port: int):
        super().__init__()
        self._url = "mysql+mysqlconnector://{}:{}@{}:{}/{}".format(user, password, host, port, db_name)

        # "mysql+mysqlconnector://<user>:<password>@<host>[:<port>]/<dbname>" # OFFICIAL BUT NOT WELL SUPPORTED CONNECTOR
        # "mysql+mysqldb://<user>:<password>@<host>[:<port>]/<dbname>"        # SLOWER BUT SUPPORTED
        # "mysql+pymysql://<username>:<password>@<host>/<dbname>[?<options>]" # VERY SLOW (PURE PYTHON)
 

class DBInterface:

    def __init__(self, url: str, echo: bool = True):
        self._engine = create_engine(url, echo=echo, future=True)
        session_factory = sessionmaker(bind=self._engine)
        # this is for multithreading (https://docs.sqlalchemy.org/en/20/orm/contextual.html)
        self._session_maker = scoped_session(session_factory)

    def create_db(self):
        if not database_exists(self._engine.url):
            create_database(self._engine.url)

    def create_all(self):
        Base.metadata.create_all(self._engine)
    
    def drop_all(self):
        Base.metadata.drop_all(self._engine)

    @property
    def engine(self):
        return self._engine

    @property
    def Session(self):
        return self._session_maker