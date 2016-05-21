#!/usr/bin/python
# coding=utf-8
import hashlib
import logging
import os
from datetime import datetime

import numpy as np
from scipy.sparse.construct import hstack
from scipy.sparse.coo import coo_matrix
from scipy.sparse.csr import csr_matrix
from scipy.sparse.dok import dok_matrix
from sklearn.externals import joblib
from sqlalchemy.orm import joinedload
from sqlalchemy.sql.expression import text

from model import DB
from model.objects.Commit import Commit
from model.objects.Repository import Repository


class Dataset:
    def __init__(self, total_feature_count, version_count, feature_list, target_id, start, end,
                 ngram_sizes=None, ngram_levels=None, label="", sparse=False, dok=False):
        """" Initialize an empty dataset.

        A dataset consists of two components:
        - The data attribute is a matrix containing all input data. It's size is version_count x feature_count.
            Each row of the data matrix represents the feature vector of one version.
        - The target attribute is a vector containing the ground truth. It's size is version_count.

        Args:
            total_feature_count (int): Amount of versions (and ngrams). Equals the rows of the data and target matrix.
            feature_list (List[str]): A list of Feature IDs. Must be in the same order as they are in the dataset.
            target_id (str): ID of the target which is used in this dataset. E.g. 'month'
            start (datetime): Start of the date range contained in this dataset.
            end (datetime): End of the date range contained in this dataset.
            ngram_sizes (list[int]): Optional. The ngram-sizes in this dataset (e.g. [1, 2] for 1-grams and 2-grams)
            ngram_levels (list[int]): Optional. The ngram-levels in this dataset.
            label (str): An arbitrary label, e.g. "Test", for this dataset. Useful when caching!
            sparse (bool): If the data and target matrices should be sparse. Recommended in combination with ngrams.
            dok (bool): If a dok-type sparse matrix should be used. Dok is faster to update. Can be converted to CSR.
        """
        ngram_count = 0
        if ngram_sizes and ngram_levels:
            ngram_count = len(ngram_sizes) * len(ngram_levels)
        logging.debug("Initializing Dataset with  %i versions, %i features and %i ngram vectors." % (
            version_count, total_feature_count, ngram_count))

        dimension = (version_count, total_feature_count + ngram_count)
        if sparse:
            if dok:
                self.data = dok_matrix(dimension, dtype=np.float64)
            else:
                self.data = csr_matrix(dimension, dtype=np.float64)
        else:
            self.data = np.zeros(dimension)
        self.target = np.zeros(version_count)
        self.feature_list = feature_list
        self.target_id = target_id
        self.start = start
        self.end = end
        self.ngram_sizes = ngram_sizes
        self.ngram_levels = ngram_levels
        self.label = label
        self.sparse = sparse

    def has_ngrams(self):
        """ True if this dataset contains ngrams. Must have at least one ngram size and level."""
        return self.ngram_sizes and self.ngram_levels

    def to_csr(self):
        """ Converts the sparse data matrix to CSR. This is more efficient to calculate with. """
        if self.sparse and type(self.data) != csr_matrix and hasattr(self.data, 'tocsr'):
            logging.debug("Converting data matrix to CSR Matrix")
            self.data.tocsr()

    def to_dok(self):
        """ Converts the sparse data matrix to DOK. This is more efficient to update or construct. """
        if self.sparse  and type(self.data) != csr_matrix and hasattr(self.data, 'todok'):
            logging.debug("Converting data matrix to DOK Matrix")
            self.data.todok()


def get_dataset(repository, start, end, feature_list, target_id, ngram_sizes=None, ngram_levels=None, label="",
                cache=False, cache_directory=None, eager_load=False, sparse=False):
    """ Reads a dataset from a repository in a specific time range.

    If cache=True, the dataset will be read from a file, if one exists. If not, after reading from the DB, it will
    be saved to a *.dataset file.
    For NGrams to be loaded into the dataset, there must be at least one ngram size and level specified.

    Args:
        repository (Repository): The repository to query. Can also be its name as a string
        start (datetime): The start range for the dataset
        end (datetime): The end range for the dataset
        feature_list (list[str]): A list of the feature-IDs to be read into the dataset.
        target_id (str): The ID of the target. Use a TARGET_X constant from UpcomingBugsForVersion
        ngram_sizes (list[int]): Optional. The ngram-sizes to be loaded in the set (e.g. [1, 2] for 1-grams and 2-grams)
        ngram_levels (list[int]): Optional. The ngram-levels to be loaded in the dataset.
        label (str): The label to be assigned to the dataset.
        cache (bool): If True, caching will be used.
        cache_directory (bool): Optional. The directory path for the cache files. If None, the working dir will be used.
        eager_load (bool): If true, all data will be loaded eagerly. This reduces database calls, but uses a lot of RAM.
        sparse (bool): If the data and target matrices should be sparse. Recommended in combination with ngrams.

    Returns:
        Dataset: The populated dataset.
    """
    if cache and not cache_directory:
        cache_directory = os.getcwd()
    if cache:
        dataset = load_dataset_file(cache_directory, label, feature_list, target_id, start, end, ngram_sizes,
                                    ngram_levels, sparse=sparse)
        if dataset is not None:
            return dataset

    dataset = get_dataset_from_db(repository, start, end, feature_list, target_id, ngram_sizes, ngram_levels,
                                  label=label, eager_load=eager_load, sparse=sparse)

    if dataset is not None and cache:
        save_dataset_file(dataset, cache_directory)

    return dataset


def get_dataset_from_db(repository, start, end, feature_list, target_id, ngram_sizes=None, ngram_levels=None, label="",
                        eager_load=False, sparse=False):
    """ Reads a dataset from a repository in a specific time range

    Args:
        repository (Repository): The repository to query. Can also be its name as a string
        start (datetime): The start range
        end (datetime): The end range
        feature_list (list[str]): A list of the feature-IDs to be read into the dataset.
        target_id (str): The ID of the target. Use a TARGET_X constant from UpcomingBugsForVersion
        ngram_sizes (list[int]): Optional. The ngram-sizes to be loaded in the set (e.g. [1, 2] for 1-grams and 2-grams)
        ngram_levels (list[int]): Optional. The ngram-levels to be loaded in the dataset.
        label (str): The label to be assigned to the dataset.
        eager_load (bool): If true, all data will be loaded eagerly. This reduces database calls, but uses a lot of RAM.
        sparse (bool): If the data and target matrices should be sparse. Recommended in combination with ngrams.

    Returns:
        Dataset: The populated dataset.
    """
    if ngram_sizes and type(ngram_sizes) != list:
        ngram_sizes = [ngram_sizes]
    if ngram_levels and type(ngram_levels) != list:
        ngram_sizes = [ngram_levels]
    use_ngrams = True if ngram_sizes and ngram_levels else False

    session = DB.create_session()

    if type(repository) is str:
        repository_name = repository
        repository = get_repository_by_name(session, repository_name)
        if repository is None:
            logging.error("Repository with name %s not found! Returning no Dataset" % repository_name)
            return None

    commits = get_commits_in_range(session, repository, start, end,
                                   eager_load_ngrams=use_ngrams and eager_load,
                                   eager_load_features=eager_load)
    if commits is None:
        logging.error("Could not retrieve commits! Returning no Dataset")
        return None
    logging.debug("Commits received.")

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
        ngrams = get_ngram_vector_list(versions[0], ngram_sizes, ngram_levels)
        ngram_count = sum([ngram.vector_size for ngram in ngrams])
        logging.debug("Ngram sizes %s and levels %s amount to %i total ngrams." % (
            str(ngram_sizes), str(ngram_levels), ngram_count))

    dataset = Dataset(feature_count + ngram_count, len(versions), feature_list, target_id, start, end, ngram_sizes,
                      ngram_levels, label, sparse=sparse, dok=True)
    i = 0
    for version in versions:
        if len(version.upcoming_bugs) == 0:
            raise Exception("Version %s has no upcoming_bugs entry. Can't retrieve target!" % version.id)
        target = version.upcoming_bugs[0].get_target(target_id)
        if target is None:
            raise Exception("Upcoming_bugs entry of Version %s has no target %s!" % (version.id, target))
        dataset.target[i] = target

        j = 0
        for feature_value in version.feature_values:
            if feature_value.feature_id in feature_list:
                dataset.data[i, j] = feature_value.value
                j += 1
        if use_ngrams:
            for ngram_vector in get_ngram_vector_list(version, ngram_sizes, ngram_levels):
                for ngram_value in ngram_vector.ngram_values.split(','):
                    dataset.data[i, j] = int(ngram_value)
                    j += 1

        if i % 100 == 0:
            logging.info("{0:.2f}% of versions processed.".format(i / len(versions) * 100))

        i += 1
    logging.info("All versions processed.")

    if sparse:
        dataset.to_csr()

    session.close()
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


def get_commits_in_range(session, repository, start, end, eager_load_features=False, eager_load_ngrams=False):
    """ Retrieves Commits with eagerly loaded versions, feature_values and upcoming_bugs.

    Args:
        session (Session): The DB-Session to use.
        repository (Repository): The repository from which to retrieve the commits.
        start (datetime): The earliest commit to retrieve.
        end (datetime): The latest commit to retrieve.
        eager_load_features (bool): If feature values should be joined eagerly. Might use a lot of memory
        eager_load_ngrams (bool): If ngram values should be joined eagerly. Might use a lot of memory

    Returns:
        List[Commit] A list of commits. None, if something went wrong.
    """
    assert start < end, "The range start must be before the range end!"
    logging.debug("Querying for Commits in repository with id %s and between %s and %s" % (repository.id, start, end))
    try:
        query = session.query(Commit). \
            options(joinedload(Commit.versions)). \
            options(joinedload('versions.file')). \
            options(joinedload('versions.upcoming_bugs'))

        if eager_load_features:
            query = query.options(joinedload('versions.feature_values'))

        if eager_load_ngrams or True:
            query = query.options(joinedload('versions.ngram_vectors'))

        query = query.filter(
            text("file_1.language = 'JAVA'"),  # Pretty much faked this one, but hey it works.
            text("version_1.deleted = 0"),
            Commit.repository_id == repository.id,
            Commit.timestamp >= start,
            Commit.timestamp < end)
        logging.debug("Running query %s" % str(query))
        return query.all()
    except:
        logging.exception(
            "Could not retrieve dataset from Repository %s in range %s to %s" % (repository.name, start, end))
        return None


def _is_ngram_vector_relevant(ngram_vector, ngram_sizes, ngram_levels):
    return ngram_vector.ngram_size in ngram_sizes and ngram_vector.ngram_level in ngram_levels


def get_ngram_vector_list(version, ngram_sizes, ngram_levels):
    """ Extracts an ordered list of ngram vectors from a version. It is ordered by ngram size first, levels second.

    Args:
        version (model.objects.Version.Version): The version from which to extract the ngrams.
        ngram_sizes (list[int]): The ngram sizes which should be put into the list.
        ngram_levels (list[int]): The ngram levels which should be put into the list.
    """
    if not ngram_sizes and not ngram_levels:
        return None
    return sorted(
        filter(lambda vector: _is_ngram_vector_relevant(vector, ngram_sizes, ngram_levels), version.ngram_vectors),
        key=lambda vector: (vector.ngram_size, vector.ngram_level)
    )


def hash_features(feature_list):
    """ Computes a hash string from a list of feature-IDs. """
    m = hashlib.md5()
    for feature in feature_list:
        m.update(feature.encode('utf8'))
    return m.hexdigest()


def generate_filename_for_dataset(dataset, strftime_format="%Y_%m_%d"):
    """ Generates the filename to cache a dataset. """
    return generate_filename(dataset.label, dataset.feature_list, dataset.target_id, dataset.start, dataset.end,
                             dataset.ngram_sizes, dataset.ngram_levels, dataset.sparse, strftime_format)


def generate_filename(label, feature_list, target_id, start, end, ngram_sizes, ngram_levels, sparse,
                      strftime_format="%Y_%m_%d"):
    """ Generates the filename to cache a dataset. """
    feature_hash = hash_features(feature_list)
    start_str = start.strftime(strftime_format)
    end_str = end.strftime(strftime_format)
    if ngram_levels and ngram_sizes:
        ngram_str = "ngrams_" + "-".join([str(x) for x in ngram_sizes]) + "_" + "-".join([str(x) for x in ngram_levels])
    else:
        ngram_str = "ngrams_no"
    sparse_str = "sparse" if sparse else "dense"
    return "_".join(
        [label, start_str, end_str, target_id, ngram_str, sparse_str, feature_hash]) + ".dataset"


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

    logging.info("Saving dataset %s to path %s." % (dataset.label, filepath))
    if dataset.sparse:
        concatenated = hstack([dataset.data, dataset.target[np.newaxis].T])
        save_sparse_matrix(concatenated, filepath)
    else:
        concatenated = np.concatenate((dataset.data, dataset.target[np.newaxis].T), axis=1)
        save_dense_matrix(concatenated, filepath, header)
    logging.debug("Saving successful")


def save_dense_matrix(data, filepath, header=""):
    np.savetxt(filepath, data, header=header)


def load_dense_matrix(filepath):
    return np.loadtxt(filepath)


def save_sparse_matrix(data, filepath):
    joblib.dump(data, filepath)


def load_sparse_matrix(filepath):
    return joblib.load(filepath)


def load_dataset_file(directory, label, feature_list, target_id, start, end, ngram_sizes, ngram_levels,
                      strftime_format="%Y_%m_%d", sparse=False):
    """ Load a dataset from a cache file.

    Args:
        directory (str): The directory in which the file should be located.
        label (str): The label for the dataset.
        feature_list (list[str]): The list of feature-IDs the dataset should contain.
        target_id (str): The ID of the target the dataset should contain.
        start (datetime): The start of the range the dataset should contain.
        end (datetime): The end of the range the dataset should contain.
        ngram_sizes (list[int]): Optional. The ngram-sizes in this dataset (e.g. [1, 2] for 1-grams and 2-grams)
        ngram_levels (list[int]): Optional. The ngram-levels in this dataset.
        strftime_format (str): Optional. The datetime string format.
        sparse (bool): If the data and target matrices should be sparse. Recommended in combination with ngrams.

    Returns:
        Dataset: The dataset, if one was retrieved. Otherwise None.
    """
    filename = generate_filename(label, feature_list, target_id, start, end, ngram_sizes, ngram_levels, sparse,
                                 strftime_format)
    filepath = os.path.join(directory, filename)
    logging.debug("Attempting to load cached dataset from %s" % filepath)
    if os.path.isfile(filepath):

        if sparse:
            concatenated = load_sparse_matrix(filepath)
            if type(concatenated) == coo_matrix:
                concatenated = concatenated.tocsr()
            data, target = concatenated[:, :-1], concatenated[:, -1]
            target = target.todense()
        else:
            concatenated = load_dense_matrix(filepath)
            data, target = np.hsplit(concatenated, [-1])
        target = np.squeeze(np.asarray(target))

        logging.debug(
            "Successfully retrieved data %s and target %s from cache file." % (str(data.shape), str(target.shape)))
        dataset = Dataset(data.shape[1], data.shape[0], feature_list, target_id, start, end, ngram_sizes, ngram_levels,
                          label, sparse=sparse)
        dataset.data = data
        dataset.target = target
        return dataset
    logging.debug("Cached dataset not found.")
    return None
