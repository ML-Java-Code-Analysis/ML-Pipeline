#!/usr/bin/python
# coding=utf-8
from sqlalchemy import Column, String, Integer, DateTime
from sqlalchemy.sql.schema import ForeignKey

from model.objects.Base import Base

Base = Base().base


class UpcomingBugsForVersion(Base):
    __tablename__ = 'upcoming_bugs_for_versions_mv'

    version_id = Column(String(36), ForeignKey('version.id'), primary_key=True, nullable=False)
    language = Column(String(20))
    path = Column(String(501), nullable=False)
    commit_id = Column(String(40), ForeignKey('commit.id'), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    month_bugs = Column(Integer, nullable=False)
    sixmonth_bugs = Column(Integer, nullable=False)
    year_bugs = Column(Integer, nullable=False)
