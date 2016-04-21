#!/usr/bin/python
# coding=utf-8
from datetime import datetime

import logging

from model import DB
from model.objects.Repository import Repository
from model.objects.Version import Version
from model.objects.Commit import Commit
import numpy as np


class Dataset:
    def __init__(self, feature_count, version_count):
        """" Initialize an empty dataset

        Args:
            feature_count: Amount of features. Equals the columns of the data matrix.
            version_count: Amount of versions. Equals the rows of the data matrix and target vector.
        """

        self.data = np.zeros((version_count, feature_count))
        self.target = np.zeros(version_count)


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
        logging.exception(
            "Could not retrieve dataset from Repository %s in range %s to %s" % (repository.name, start, end))
        return None

    version_count = 0
    for commit in commits:
        version_count += len(commit.versions)
    logging.debug("%i commits with %i versions found." % (len(commits), version_count))

    # TODO: Maybe determine size of Feature matrix beforehand and initialize it
    # TODO: It's suuuuper slow.... maybe output a progression
    dataset = None
    i = 0
    for commit in commits:
        for version in commit.versions:
            if dataset is None:
                dataset = Dataset(len(version.feature_values), version_count)

            # TODO: Handle cases where not all features are present (i.e. discard this version)
            # TODO: make target configurable
            dataset.target[i] = version.upcoming_bugs[0].sixmonth_bugs
            j = 0
            for feature_value in version.feature_values:
                # TODO: Implement a filter, where one can in/exclude features (via config)
                dataset.data[i][j] = feature_value.value()
                j += 1
            i += 1
    print(dataset)
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
