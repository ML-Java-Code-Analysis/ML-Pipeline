#!/usr/bin/python
# coding=utf-8
from sqlalchemy import Column, String, Integer
from sqlalchemy.sql.schema import ForeignKey

from model.objects.Base import Base

Base = Base().base


# noinspection PyClassHasNoInit
class NGramVector(Base):
    __tablename__ = 'ngram_vector'

    version_id = Column(String(36), ForeignKey('version.id'), primary_key=True)
    ngram_size = Column(Integer, primary_key=True)
    ngram_level = Column(Integer, primary_key=True)
    vector_size = Column(Integer)
    ngram_values = Column(String)
