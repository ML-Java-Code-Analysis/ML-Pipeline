#!/usr/bin/python
# coding=utf-8
import logging

from sklearn import linear_model
from sklearn import preprocessing

from ml.Report import Report

MODEL_TYPE_LINREG = 'LINEAR_REGRESSION'
MODEL_TYPE_RIDREG = 'RIDGE_REGRESSION'


def create_model(model_type, normalize=False):
    """ Creates a new model of the specified type.

    Args:
        model_type (str): The type of model to create. Use one of the MODEL_TYPE_X constants.
        normalize (bool): If normalization is to be used.

    Returns:
        (LinearModel) The model instance.
    """
    model_type = model_type.upper()
    logging.debug("Creating model with type %s" % model_type)
    if model_type == MODEL_TYPE_LINREG:
        return linear_model.LinearRegression(
            fit_intercept=True,
            normalize=normalize,
            copy_X=True,
        )
    elif model_type == MODEL_TYPE_RIDREG:
        return linear_model.Ridge(
            alpha=1,
            fit_intercept=True,
            normalize=normalize,
            copy_X=True,
        )
    else:
        raise ValueError("The model type %s is not supported." % model_type)


def train_model(model, train_dataset, polynomial_degree=1):
    """ Trains a model.

    Args:
        model (str|LinearModel): Model instance or string identifying the model type.
            If the latter, use of the MODEL_TYPE_X constants.
        train_dataset (Dataset): The dataset to train the model with.

    Returns:
        (LinearModel) The model instance.
    """
    if type(model) == str:
        model = create_model(model)
    if polynomial_degree > 1:
        logging.debug("Preprocessing: Generate polynomial features")
        preprocessing.PolynomialFeatures(degree=polynomial_degree, )

    logging.debug("Fitting training set to model")
    model.fit(train_dataset.data, train_dataset.target)
    return model

'''
def test_model(model, test_dataset):
    """ Test the trained model with the given dataset.

    Args:
        model (LinearModel): A trained Model instance.
        test_dataset (Dataset): The test dataset, including test input as well as target ground truth.
    """
    # logging.debug("Testing model with %s dataset" % test_dataset.size)
    target = test_dataset.target
    predicted = model.predict(test_dataset.data)

    return Report(target, predicted)
'''