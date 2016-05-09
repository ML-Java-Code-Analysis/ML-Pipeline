# coding=utf-8
from sqlalchemy import Column, String, Integer
from sqlalchemy.orm import relationship
from sqlalchemy.sql.schema import ForeignKey

from model.objects.Base import Base
from model.objects.CommitIssue import CommitIssue

Base = Base().base

TYPE_BUG = 'BUG'
TYPE_ENHANCEMENT = 'ENHANCEMENT'
TYPE_OTHER = 'OTHER'


# noinspection PyClassHasNoInit
class Issue(Base):
    __tablename__ = 'issue'

    id = Column(String(20), primary_key=True)
    issue_tracking_id = Column(Integer, ForeignKey("issueTracking.id"), primary_key=True)
    title = Column(String(500))
    type = Column(String(20), nullable=False)
    #commits = relationship("Commit", secondary=CommitIssue.__table__, back_populates="issues")
