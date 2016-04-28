#!/usr/bin/python
# coding=utf-8
import logging
import os
from datetime import datetime

import numpy as np
from sqlalchemy.orm import joinedload

from model import DB
from model.objects.Commit import Commit
from model.objects.Repository import Repository
import hashlib


class Dataset:
    def __init__(self, feature_count, version_count, feature_list, target_id, start, end, label=""):
        """" Initialize an empty dataset.

        A dataset consists of two components:
        - The data attribute is a matrix containing all input data. It's size is version_count x feature_count.
            Each row of the data matrix represents the feature vector of one version.
        - The target attribute is a vector containing the ground truth. It's size is version_count.

        Args:
            feature_count (int): Amount of features. Equals the columns of the data matrix.
            version_count (int): Amount of versions. Equals the rows of the data matrix and target vector.
            feature_list (List[str]): A list of Feature IDs. Must be in the same order as they are in the dataset.
            start (datetime): Start of the date range contained in this dataset.
            end (datetime): End of the date range contained in this dataset.
        """
        logging.debug("Initializing Dataset with %i features and %i versions." % (feature_count, version_count))
        self.data = np.zeros((version_count, feature_count))
        self.target = np.zeros(version_count)
        self.feature_list = feature_list
        self.target_id = target_id
        self.start = start
        self.end = end
        self.label = label


def get_dataset(repository, start, end, feature_list, target_id, label="", cache=False, cache_directory=None):
    """ Reads a dataset from a repository in a specific time range

    Args:
        repository (Repository): The repository to query. Can also be its name as a string
        start (datetime): The start range
        end (datetime): The end range
    """
    if cache and not cache_directory:
        cache_directory = os.getcwd()
    if cache:
        dataset = load_dataset_file(cache_directory, label, feature_list, target_id, start, end)
        if dataset is not None:
            return dataset

    dataset = get_dataset_from_db(repository, start, end, feature_list, target_id, label=label)

    if cache:
        save_dataset_file(dataset, cache_directory)

    return dataset


def get_dataset_from_db(repository, start, end, feature_list, target_id, label=""):
    """ Reads a dataset from a repository in a specific time range

    Args:
        repository (Repository): The repository to query. Can also be its name as a string
        start (datetime): The start range
        end (datetime): The end range
    """
    """if cache:
        if not cache_directory:
            cache_directory = os.getcwd()
        if dataset_file_exists(start, end, feature_list, cache_directory):
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
    feature_count = len(feature_list)
    logging.debug("%i features found." % feature_count)

    dataset = Dataset(feature_count, version_count, feature_list, target_id, start, end, label)
    i = 0
    for commit in commits:
        for version in commit.versions:

            if len(version.upcoming_bugs) == 0:
                logging.warning(
                    "Version %s has no upcoming_bugs entry. Can't retrieve target, skipping version." % version.id)
                continue
            target = version.upcoming_bugs[0].get_target(target_id)
            if target is None:
                logging.warning("Upcoming_bugs entry of Version %s has no target %s. skipping version." % (
                    version.id, target))
                continue
            dataset.target[i] = target
            j = 0
            for feature_value in version.feature_values:
                if feature_value.feature_id in feature_list:
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


def hash_features(feature_list):
    m = hashlib.md5()
    for feature in feature_list:
        m.update(feature.encode('utf8'))
    return m.hexdigest()


def generate_filename_for_dataset(dataset, strftime_format="%Y_%m_%d"):
    return generate_filename(dataset.label, dataset.feature_list, dataset.target_id, dataset.start, dataset.end,
                             strftime_format)


def generate_filename(label, feature_list, target_id, start, end, strftime_format="%Y_%m_%d"):
    feature_hash = hash_features(feature_list)
    start_str = start.strftime(strftime_format)
    end_str = end.strftime(strftime_format)
    return "_".join([label, start_str, end_str, target_id, feature_hash]) + ".dataset"


def get_file_header(dataset):
    return ",".join(dataset.feature_list)


def read_file_header(header, strftime_format="%y_%m_%d"):
    """

    Args:
        header (str):
        strftime_format (str):
    """
    fragments = header.split("\n")
    start = datetime.strptime(fragments[0], strftime_format)
    end = datetime.strptime(fragments[1], strftime_format)
    feature_list = fragments[2].split(",")
    return start, end, feature_list


def save_dataset_file(dataset, directory):
    filepath = os.path.join(directory, generate_filename_for_dataset(dataset))
    header = get_file_header(dataset)
    concatenated_array = np.concatenate((dataset.data, dataset.target[np.newaxis].T), axis=1)

    logging.info("Saving dataset %s to path %s." % (dataset.label, filepath))
    np.savetxt(filepath, concatenated_array, header=header)
    logging.debug("Saving successful")


def load_dataset_file(directory, label, feature_list, target_id, start, end, strftime_format="%Y_%m_%d"):
    filename = generate_filename(label, feature_list, target_id, start, end, strftime_format)
    filepath = os.path.join(directory, filename)
    logging.debug("Attempting to load cached dataset from %s" % filepath)
    if os.path.isfile(filepath):
        concatenated_array = np.loadtxt(filepath)

        data, target = np.hsplit(concatenated_array, [-1])

        logging.debug(
            "Successfully retrieved data %s and target %s from cache file." % (str(data.shape), str(target.shape)))
        dataset = Dataset(data.shape[0], data.shape[0], feature_list, target, start, end, label)
        dataset.data = data
        dataset.target = target
        return dataset
    logging.debug("Cached dataset not found.")
    return None
