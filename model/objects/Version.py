# coding=utf-8
from sqlalchemy import Column, Integer, String, Boolean
from sqlalchemy.orm import relationship
from sqlalchemy.sql.schema import ForeignKey

from model.objects.Base import Base

Base = Base().base

MAX_PATH_LENGTH = 500


# noinspection PyClassHasNoInit
class Version(Base):
    __tablename__ = 'version'

    id = Column(String(36), primary_key=True)
    file_id = Column(String(36), ForeignKey('file.id'))
    commit_id = Column(String(40), ForeignKey('commit.id'))
    # path = Column(String(MAX_PATH_LENGTH + 1), nullable=False)
    # lines_added = Column(Integer, nullable=False)
    # lines_deleted = Column(Integer, nullable=False)
    # file_size = Column(Integer)
    # deleted = Column(Boolean)
    # lines = relationship('Line')
    feature_values = relationship('FeatureValue')
    ngram_vectors = relationship('NGramVector')
    upcoming_bugs = relationship('UpcomingBugsForVersion')
