# coding=utf-8
from sqlalchemy import Column, String, Integer
from sqlalchemy.orm import relationship
from sqlalchemy.sql.schema import ForeignKey

from model.objects.Base import Base

Base = Base().base

TYPE_GITHUB = 'GITHUB'
TYPE_JIRA = 'JIRA'


# noinspection PyClassHasNoInit
class IssueTracking(Base):
    __tablename__ = "issueTracking"

    id = Column(Integer, primary_key=True)
    repository_id = Column(Integer, ForeignKey('repository.id'))
    repository = relationship("Repository", back_populates='issueTracking')
    issues = relationship("Issue")
    type = Column(String(20), nullable=False)
    url = Column(String(200), nullable=False)
    username = Column(String(100))
    password = Column(String(100))
