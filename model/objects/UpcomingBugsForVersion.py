#!/usr/bin/python
# coding=utf-8
from sqlalchemy import Column, String, Integer, DateTime
from sqlalchemy.sql.schema import ForeignKey

from model.objects.Base import Base

Base = Base().base

TARGET_BUGS_MONTH = 'month'
TARGET_BUGS_SIXMONTHS = 'sixmonths'
TARGET_BUGS_YEAR = 'year'


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

    def get_target(self, target):
        if target == TARGET_BUGS_MONTH:
            return self.month_bugs
        elif target == TARGET_BUGS_SIXMONTHS:
            return self.sixmonth_bugs
        elif target == TARGET_BUGS_YEAR:
            return self.year_bugs
