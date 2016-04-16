#!/usr/bin/python
# coding=utf-8
from sqlalchemy import Column, String, Float
from sqlalchemy.sql.schema import ForeignKey

from model.objects.Base import Base

Base = Base().base


# noinspection PyClassHasNoInit
class FeatureValue(Base):
    __tablename__ = 'feature_value'

    feature_id = Column(String(36), primary_key=True)
    version_id = Column(String(36), ForeignKey('version.id'), primary_key=True)
    value = Column(Float)
