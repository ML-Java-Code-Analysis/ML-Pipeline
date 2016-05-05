#!/usr/bin/python
# coding=utf-8
import logging
import numpy as np
from sklearn import preprocessing


def preprocess(dataset, polynomial_degree=1):
    """

    Args:
        dataset (ml.Dataset.Dataset): The data to preprocess
        polynomial_degree (int): If higher than 1, polynomial features will be used.
    """
    data = np.copy(dataset.data)
    if polynomial_degree > 1:
        logging.debug("Preprocessing: Generate polynomial features")
        poly = preprocessing.PolynomialFeatures(degree=polynomial_degree)
        data = poly.fit_transform(data)
    return data
