#!/usr/bin/python
# coding=utf-8
from datetime import datetime

import logging

from model import DB
from model.objects.Repository import Repository
from model.objects.Version import Version
from model.objects.Commit import Commit


def get_dataset_from_range(repository, start, end):
    """ Reads a dataset from a repository in a specific time range

    Args:
        repository (Repository): The repository to query. Can also be its name as a string
        start (datetime): The start range
        end (datetime): The end range
    """
    dataset = None
    session = DB.create_session()

    if type(repository) is str:
        repository_name = repository
        repository = get_repository_by_name(repository_name, session)
        if repository is None:
            logging.error("Repository with name %s not found! Returning no Dataset" % repository_name)
            return None

    logging.debug("Querying for Commits in repository with id %s and between %s and %s" % (repository.id, start, end))
    try:
        query = session.query(Commit).filter(
            Commit.repository_id == repository.id,
            Commit.timestamp >= start,
            Commit.timestamp < end)

        commits = query.all()
    except:
        logging.exception("Could not retrieve dataset from Repository %s in range %s to %s" % (repository.name, start, end))
        return None
    logging.debug("%s commits found. Creating Dataset Array" % len(commits))

    for commit in commits:
        for version in commit.versions:
            for feature_value in version.feature_values:
                pass # TODO: build dataset array



    session.close()
    return dataset


def get_repository_by_name(name, session):
    """ Retrieves a repository from the DB by its name

    Args:
        name (str): The name of the repository

    Returns:
        (Repository) the repository if it was found, or None
    """
    repository = None
    logging.debug("Retrieving repository with name %s" % name)
    try:
        query = session.query(Repository).filter(Repository.name == name)
        repository = query.one_or_none()
    except Exception as e:
        logging.exception("Repository with name %s could not be retrieved" % name)
    return repository
