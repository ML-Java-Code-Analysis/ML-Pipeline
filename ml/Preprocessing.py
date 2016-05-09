#!/usr/bin/python
# coding=utf-8
import logging
from sklearn import preprocessing


def preprocess(dataset, normalize=False, polynomial_degree=1):
    """

    Args:
        dataset (ml.Dataset.Dataset): The data to preprocess
        polynomial_degree (int): If higher than 1, polynomial features will be used.
    """
    if normalize:
        normalize_data(dataset)
    if polynomial_degree > 1:
        polynomial_features(dataset, polynomial_degree)
    return dataset


def normalize_data(dataset):
    logging.debug("Preprocessing: Normalizing data")
    scaler = preprocessing.StandardScaler()
    dataset.data = scaler.fit_transform(dataset.data)
    return dataset


def polynomial_features(dataset, polynomial_degree):
    logging.debug("Preprocessing: Generate polynomial features")
    poly = preprocessing.PolynomialFeatures(degree=polynomial_degree)
    dataset.data = poly.fit_transform(dataset.data)
    return dataset
