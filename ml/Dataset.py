#!/usr/bin/python
# coding=utf-8
import hashlib
import logging
import os
from datetime import datetime

import numpy as np
from sqlalchemy.orm import joinedload

from model import DB
from model.objects.Version import Version
from model.objects.File import File
from model.objects.Commit import Commit
from model.objects.Repository import Repository


class Dataset:
    def __init__(self, feature_count, version_count, feature_list, target_id, start, end, has_ngrams, label=""):
        """" Initialize an empty dataset.

        A dataset consists of two components:
        - The data attribute is a matrix containing all input data. It's size is version_count x feature_count.
            Each row of the data matrix represents the feature vector of one version.
        - The target attribute is a vector containing the ground truth. It's size is version_count.

        Args:
            feature_count (int): Amount of features. Equals the columns of the data matrix.
            version_count (int): Amount of versions. Equals the rows of the data matrix and target vector.
            feature_list (List[str]): A list of Feature IDs. Must be in the same order as they are in the dataset.
            target_id (str): ID of the target which is used in this dataset. E.g. 'month'
            start (datetime): Start of the date range contained in this dataset.
            end (datetime): End of the date range contained in this dataset.
            label (str): An arbitrary label, e.g. "Test", for this dataset. Useful when caching!
        """
        logging.debug("Initializing Dataset with %i features and %i versions." % (feature_count, version_count))
        self.data = np.zeros((version_count, feature_count))
        self.target = np.zeros(version_count)
        self.feature_list = feature_list
        self.target_id = target_id
        self.start = start
        self.end = end
        self.has_ngrams = has_ngrams
        self.label = label


def get_dataset(repository, start, end, feature_list, target_id, use_ngrams=False, label="", cache=False,
                cache_directory=None):
    """ Reads a dataset from a repository in a specific time range.

    If cache=True, the dataset will be read from a file, if one exists. If not, after reading from the DB, it will
    be saved to a *.dataset file.

    Args:
        repository (Repository): The repository to query. Can also be its name as a string
        start (datetime): The start range for the dataset
        end (datetime): The end range for the dataset
        feature_list (list[str]): A list of the feature-IDs to be read into the dataset.
        target_id (str): The ID of the target. Use a TARGET_X constant from UpcomingBugsForVersion
        label (str): The label to be assigned to the dataset.
        cache (bool): If True, caching will be used.
        cache_directory (bool): Optional. The directory path for the cache files. If None, the working dir will be used.

    Returns:
        Dataset: The populated dataset.
    """
    if cache and not cache_directory:
        cache_directory = os.getcwd()
    if cache:
        dataset = load_dataset_file(cache_directory, label, feature_list, target_id, start, end, use_ngrams)
        if dataset is not None:
            return dataset

    dataset = get_dataset_from_db(repository, start, end, feature_list, target_id, use_ngrams,
                                  label=label)

    if dataset is not None and cache:
        save_dataset_file(dataset, cache_directory)

    return dataset


def get_dataset_from_db(repository, start, end, feature_list, target_id, use_ngrams=False, ngram_sizes=None,
                        ngram_levels=None, label=""):
    """ Reads a dataset from a repository in a specific time range

    Args:
        repository (Repository): The repository to query. Can also be its name as a string
        start (datetime): The start range
        end (datetime): The end range
        feature_list (list[str]): A list of the feature-IDs to be read into the dataset.
        target_id (str): The ID of the target. Use a TARGET_X constant from UpcomingBugsForVersion
        label (str): The label to be assigned to the dataset.

    Returns:
        Dataset: The populated dataset.
    """
    session = DB.create_session()

    if type(repository) is str:
        repository_name = repository
        repository = get_repository_by_name(session, repository_name)
        if repository is None:
            logging.error("Repository with name %s not found! Returning no Dataset" % repository_name)
            return None

    commits = get_commits_in_range(session, repository, start, end, use_ngrams=use_ngrams)
    if commits is None:
        logging.error("Could not retrieve commits! Returning no Dataset")
        return None
    logging.debug("Commits received.")
    session.close()

    if len(commits) == 0:
        logging.error("No Commits found!")
        return None

    versions = []
    for commit in commits:
        versions += commit.versions
    logging.debug("%i commits with %i versions found." % (len(commits), len(versions)))

    feature_count = len(feature_list)
    logging.debug("%i features found." % feature_count)

    ngram_count = 0
    if use_ngrams:
        for ngram_vector in versions[0].ngram_vectors:
            if ngram_sizes is None or ngram_vector.ngram_size in ngram_sizes \
                    and ngram_levels is None or ngram_vector.ngram_level in ngram_levels:
                ngram_count += ngram_vector.vector_size
        logging.debug("%i total ngrams." % ngram_count)

    dataset = Dataset(feature_count + ngram_count, len(versions), feature_list, target_id, start, end, use_ngrams,
                      label)
    i = 0
    for version in versions:
        if len(version.upcoming_bugs) == 0:
            raise Exception("Version %s has no upcoming_bugs entry. Can't retrieve target!" % version.id)
        target = version.upcoming_bugs[0].get_target(target_id)
        if target is None:
            raise Exception("Upcoming_bugs entry of Version %s has no target %s!" % (version.id, target))
        dataset.target[i] = target
        j = 0
        print(version.id + "\t" + str([x.feature_id for x in version.feature_values]))
        for feature_value in version.feature_values:
            if feature_value.feature_id in feature_list:
                dataset.data[i][j] = feature_value.value
                j += 1
        for ngram_vector in version.ngram_vectors:
            if ngram_sizes is None or ngram_vector.ngram_size in ngram_sizes \
                    and ngram_levels is None or ngram_vector.ngram_level in ngram_levels:
                pass
        i += 1

    return dataset


def get_repository_by_name(session, name):
    """ Retrieves a repository from the DB by its name

    Args:
        session (Session): The DB-Session to use.
        name (str): The name of the repository

    Returns:
        Repository: the repository if it was found, or None
    """
    repository = None
    logging.debug("Retrieving repository with name '%s'" % name)
    try:
        query = session.query(Repository).filter(Repository.name == name)
        repository = query.one_or_none()
    except:
        logging.exception("Repository with name %s could not be retrieved" % name)
    return repository


def get_commits_in_range(session, repository, start, end, use_ngrams=False):
    """ Retrieves Commits with eagerly loaded versions, feature_values and upcoming_bugs.

    Args:
        session (Session): The DB-Session to use.
        repository (Repository): The repository from which to retrieve the commits.
        start (datetime): The earliest commit to retrieve.
        end (datetime): The latest commit to retrieve.

    Returns:
        List[Commit] A list of commits. None, if something went wrong.
    """
    assert start < end, "The range start must be before the range end!"
    # TODO: Maybe filter here already for ngrams and vectors?
    # TODO: Order by feature, order by ngram size, level!
    logging.debug("Querying for Commits in repository with id %s and between %s and %s" % (repository.id, start, end))
    try:
        # Load
        # TODO: Fix this shit
        query = session.query(Commit). \
            options(joinedload(Commit.versions)). \
            options(joinedload('versions.file')). \
            options(joinedload('versions.feature_values')). \
            options(joinedload('versions.upcoming_bugs'))

        if use_ngrams:
            query = query.options(joinedload('versions.ngram_vectors'))

        query = query.filter(
            "file_1.language = 'JAVA'", # Pretty much faked this one, but hey it works.
            Commit.repository_id == repository.id,
            Commit.timestamp >= start,
            Commit.timestamp < end)
        logging.debug("Running query %s" % str(query))
        return query.all()
    except:
        logging.exception(
            "Could not retrieve dataset from Repository %s in range %s to %s" % (repository.name, start, end))
        return None


def hash_features(feature_list):
    """ Computes a hash string from a list of feature-IDs. """
    m = hashlib.md5()
    for feature in feature_list:
        m.update(feature.encode('utf8'))
    return m.hexdigest()


def generate_filename_for_dataset(dataset, strftime_format="%Y_%m_%d"):
    """ Generates the filename to cache a dataset. """
    return generate_filename(dataset.label, dataset.feature_list, dataset.target_id, dataset.start, dataset.end,
                             dataset.has_ngrams, strftime_format)


def generate_filename(label, feature_list, target_id, start, end, use_ngrams,
                      strftime_format="%Y_%m_%d"):
    """ Generates the filename to cache a dataset. """
    feature_hash = hash_features(feature_list)
    start_str = start.strftime(strftime_format)
    end_str = end.strftime(strftime_format)
    ngram_str = "ngram" if use_ngrams else "nongram"
    return "_".join(
        [label, start_str, end_str, target_id, ngram_str, feature_hash]) + ".dataset"


def get_file_header(dataset):
    """ Generates the file header to be used in the cache file. """
    return ",".join(dataset.feature_list)


def save_dataset_file(dataset, directory):
    """ Cache a dataset into a file.

    Args:
        dataset (Dataset): The dataset to save.
        directory (str): The directory to save it to.
    """
    filepath = os.path.join(directory, generate_filename_for_dataset(dataset))
    header = get_file_header(dataset)
    concatenated_array = np.concatenate((dataset.data, dataset.target[np.newaxis].T), axis=1)

    logging.info("Saving dataset %s to path %s." % (dataset.label, filepath))
    np.savetxt(filepath, concatenated_array, header=header)
    logging.debug("Saving successful")


def load_dataset_file(directory, label, feature_list, target_id, start, end, use_ngrams, strftime_format="%Y_%m_%d"):
    """ Load a dataset from a cache file.

    Args:
        directory (str): The directory in which the file should be located.
        label (str): The label for the dataset.
        feature_list (list[str]): The list of feature-IDs the dataset should contain.
        target_id (str): The ID of the target the dataset should contain.
        start (datetime): The start of the range the dataset should contain.
        end (datetime): The end of the range the dataset should contain.
        strftime_format (str): Optional. The datetime string format.

    Returns:
        Dataset: The dataset, if one was retrieved. Otherwise None.
    """
    filename = generate_filename(label, feature_list, target_id, start, end, use_ngrams,
                                 strftime_format)
    filepath = os.path.join(directory, filename)
    logging.debug("Attempting to load cached dataset from %s" % filepath)
    if os.path.isfile(filepath):
        concatenated_array = np.loadtxt(filepath)

        data, target = np.hsplit(concatenated_array, [-1])

        logging.debug(
            "Successfully retrieved data %s and target %s from cache file." % (str(data.shape), str(target.shape)))
        dataset = Dataset(data.shape[1], data.shape[0], feature_list, target_id, start, end, use_ngrams, label)
        dataset.data = data
        dataset.target = target.T[0]
        return dataset
    logging.debug("Cached dataset not found.")
    return None
