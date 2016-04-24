#!/usr/bin/python
# coding=utf-8
from datetime import datetime

import logging

from sqlalchemy.orm import joinedload

from model import DB
from model.objects.Repository import Repository
from model.objects.Commit import Commit
import numpy as np

from utils import Config


class Dataset:
    def __init__(self, feature_count, version_count):
        """" Initialize an empty dataset.

        A dataset consists of two components:
        - The data attribute is a matrix containing all input data. It's size is version_count x feature_count.
            Each row of the data matrix represents the feature vector of one version.
        - The target attribute is a vector containing the ground truth. It's size is version_count.

        Args:
            feature_count: Amount of features. Equals the columns of the data matrix.
            version_count: Amount of versions. Equals the rows of the data matrix and target vector.
        """
        logging.debug("Initializing Dataset with %i features and %i versions." % (feature_count, version_count))
        self.data = np.zeros((version_count, feature_count))
        self.target = np.zeros(version_count)


def get_dataset_from_range(repository, start, end):
    """ Reads a dataset from a repository in a specific time range

    Args:
        repository (Repository): The repository to query. Can also be its name as a string
        start (datetime): The start range
        end (datetime): The end range
    """
    session = DB.create_session()

    if type(repository) is str:
        repository_name = repository
        repository = get_repository_by_name(session, repository_name)
        if repository is None:
            logging.error("Repository with name %s not found! Returning no Dataset" % repository_name)
            return None

    commits = get_commits_in_range(session, repository, start, end)
    if commits is None:
        logging.error("Could not retrieve commits! Returning no Dataset")
    logging.debug("Commits received.")

    version_count = 0
    for commit in commits:
        version_count += len(commit.versions)
    logging.debug("%i commits with %i versions found." % (len(commits), version_count))

    # TODO: Skipping versions is kinda bad because then the dataset won't be full...
    # TODO: Maybe shorten dataset afterwards?
    feature_count = len(Config.dataset_features)
    logging.debug("%i features found." % feature_count)
    dataset = Dataset(feature_count, version_count)
    i = 0
    for commit in commits:
        for version in commit.versions:

            if len(version.upcoming_bugs) == 0:
                logging.warning(
                    "Version %s has no upcoming_bugs entry. Can't retrieve target, skipping version." % version.id)
                continue
            target = version.upcoming_bugs[0].get_target(Config.dataset_target)
            if target is None:
                logging.warning("Upcoming_bugs entry of Version %s has no target %s. skipping version." % (
                    version.id, Config.dataset_target))
                continue
            dataset.target[i] = target
            j = 0
            for feature_value in version.feature_values:
                if feature_value.feature_id in Config.dataset_features:
                    dataset.data[i][j] = feature_value.value
                    j += 1
            i += 1
    session.close()
    return dataset


def get_repository_by_name(session, name):
    """ Retrieves a repository from the DB by its name

    Args:
        session (Session): The DB-Session to use.
        name (str): The name of the repository

    Returns:
        (Repository) the repository if it was found, or None
    """
    repository = None
    logging.debug("Retrieving repository with name '%s'" % name)
    try:
        query = session.query(Repository).filter(Repository.name == name)
        repository = query.one_or_none()
    except:
        logging.exception("Repository with name %s could not be retrieved" % name)
    return repository


def get_commits_in_range(session, repository, start, end):
    """ Retrieves Commits with eagerly loaded versions, feature_values and upcoming_bugs.

    Args:
        session (Session): The DB-Session to use.
        repository (Repository): The repository from which to retrieve the commits.
        start (datetime): The earliest commit to retrieve.
        end (datetime): The latest commit to retrieve.

    Returns:
        (List[Commit]) A list of commits. None, if something went wrong.
    """
    assert start < end, "The range start must be before the range end!"
    logging.debug("Querying for Commits in repository with id %s and between %s and %s" % (repository.id, start, end))
    try:
        # Load
        query = session.query(Commit). \
            options(joinedload(Commit.versions)). \
            options(joinedload('versions.feature_values')). \
            options(joinedload('versions.upcoming_bugs')). \
            filter(
            Commit.repository_id == repository.id,
            Commit.timestamp >= start,
            Commit.timestamp < end)

        return query.all()
    except:
        logging.exception(
            "Could not retrieve dataset from Repository %s in range %s to %s" % (repository.name, start, end))
        return None
