# coding=utf-8
import logging

from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session
from sqlalchemy.orm import sessionmaker
from model.objects.Base import Base
from utils import Config

__engine = None
__Session = None


def __get_engine():
    global __engine

    db_dialect = Config.database_dialect
    db_name = Config.database_name
    user = Config.database_user
    password = Config.database_user_password
    host = Config.database_host
    port = Config.database_port

    auth_string = ""
    if user:
        auth_string += user
        if password:
            auth_string += ':' + password
        auth_string += '@'
    port_string = ""
    if port:
        port_string += ':' + str(port)

    if __engine is None:
        if db_dialect.upper() == 'SQLITE':
            __engine = create_engine('sqlite:///{0}.db'.format(db_name))
        elif db_dialect.upper() == 'MYSQL':
            url = r'mysql+pymysql://{auth_string}{host}{port_string}/{db_name}?charset=utf8'.format(
                auth_string=auth_string,
                host=host,
                port_string=port_string,
                db_name=db_name
            )
            __engine = create_engine(url, pool_recycle=3600)
        elif db_dialect.upper() == 'POSTGRES':
            url = r'postgresql+pg8000://{auth_string}{host}{port_string}/{db_name}'.format(
                auth_string=auth_string,
                host=host,
                port_string=port_string,
                db_name=db_name
            )
            __engine = create_engine(url, client_encoding='utf8')
        else:
            logging.critical("SQL Dialect " + db_dialect + " is not supported.")
            return None
    return __engine


def __get_session(engine):
    global __Session
    if __Session is None:
        session_factory = sessionmaker(bind=engine)
        __Session = scoped_session(session_factory)
    return __Session


# noinspection PyUnresolvedReferences
def create_db():
    # Import all ORM objects to register them in SQLAlchemy Base
    from model.objects.IssueTracking import IssueTracking
    from model.objects.Repository import Repository
    from model.objects.Commit import Commit
    from model.objects.File import File
    from model.objects.Issue import Issue
    from model.objects.Version import Version
    from model.objects.Line import Line

    engine = __get_engine()
    if engine is None:
        Log.error("DB Engine could not be created! DB init failed!")
    else:
        Base().base.metadata.create_all(engine)


def create_session():
    engine = __get_engine()
    if engine is None:
        logging.critical("DB Engine could not be created! Session creation failed!")
        return None

    new_session = __get_session(engine)
    if new_session is None:
        logging.error("Could not create a session!")

    return new_session
