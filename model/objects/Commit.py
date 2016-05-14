# coding=utf-8
from sqlalchemy import Column, String, Integer, DateTime, Boolean
from sqlalchemy.orm import relationship
from sqlalchemy.sql.schema import ForeignKey

from model.objects.Base import Base
from model.objects.CommitIssue import CommitIssue

Base = Base().base
MAX_MESSAGE_LENGTH = 3000
MAX_AUTHOR_LENGTH = 100


# noinspection PyClassHasNoInit
class Commit(Base):
    __tablename__ = 'commit'

    id = Column(String(40), primary_key=True)
    repository_id = Column(Integer, ForeignKey('repository.id'))
    # message = Column(String(MAX_MESSAGE_LENGTH+1), nullable=False)
    # author = Column(String(MAX_AUTHOR_LENGTH+1), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    # added_files_count = Column(Integer)
    # deleted_files_count = Column(Integer)
    # changed_files_count = Column(Integer)
    # renamed_files_count = Column(Integer)
    # project_size = Column(Integer)
    # project_file_count = Column(Integer)
    # complete = Column(Boolean)
    # issues = relationship("Issue", secondary=CommitIssue.__table__, back_populates="commits")
    versions = relationship("Version", back_populates='commit')
