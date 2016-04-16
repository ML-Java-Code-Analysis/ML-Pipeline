# coding=utf-8
from sqlalchemy.ext.declarative import declarative_base

from utils.Borg import Borg


class Base(Borg):
    metadata = None

    def __init__(self):
        Borg.__init__(self)
        if not hasattr(self, 'base'):
            self.base = declarative_base()
