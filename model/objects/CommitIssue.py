# coding=utf-8
from sqlalchemy import Column, String, Integer
from sqlalchemy.sql.schema import ForeignKey, ForeignKeyConstraint

from model.objects.Base import Base

Base = Base().base


# noinspection PyClassHasNoInit
class CommitIssue(Base):
    __tablename__ = 'commit_issue'

    commit_id = Column('commit_id', String(40), ForeignKey('commit.id'), primary_key=True)
    issue_id = Column('issue_id', String(20), primary_key=True)
    issue_tracking_id = Column('issue_tracking_id', Integer, primary_key=True)

    __table_args__ = (ForeignKeyConstraint([issue_id, issue_tracking_id],
                                           ['issue.id', 'issue.issue_tracking_id']),
                      {})
