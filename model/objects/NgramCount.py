#!/usr/bin/python
# coding=utf-8
from sqlalchemy import Column, String, Integer
from sqlalchemy.sql.schema import ForeignKey

from model.objects.Base import Base

Base = Base().base


# noinspection PyClassHasNoInit
class NGramCount(Base):
    __tablename__ = 'ngram_count'

    ngram_id = Column(String(500), primary_key=True)
    version_id = Column(String(36), ForeignKey('version.id'), primary_key=True)
    count = Column(Integer)
