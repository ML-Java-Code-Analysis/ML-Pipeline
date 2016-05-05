#!/usr/bin/python
# coding=utf-8
import logging

from sklearn import linear_model
from sklearn import svm

from ml import Preprocessing

MODEL_TYPE_LINREG = 'LINEAR_REGRESSION'
MODEL_TYPE_RIDREG = 'RIDGE_REGRESSION'
MODEL_TYPE_SVR = 'SVR'


def create_model(model_type, normalize=False, alpha=None, kernel=None):
    """ Creates a new model of the specified type.

    Args:
        model_type (str): The type of model to create. Use one of the MODEL_TYPE_X constants.
        normalize (bool): If normalization is to be used.
        alpha (float): The regularization parameter. Will only be used if applicable to the model type.
        kernel (str): The kernel to use, if applicable to the model type.

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
            alpha=alpha,
            fit_intercept=True,
            normalize=normalize,
            copy_X=True,
        )
    elif model_type == MODEL_TYPE_SVR:
        return svm.SVR(kernel=kernel)
    else:
        raise ValueError("The model type %s is not supported." % model_type)


def train_model(model, train_dataset, polynomial_degree=1):
    """ Trains a model.

    Args:
        model (str|LinearModel): Model instance or string identifying the model type.
            If the latter, use of the MODEL_TYPE_X constants.
        train_dataset (Dataset): The dataset to train the model with.
        polynomial_degree (int): If higher than 1, polynomial features will be used.
    Returns:
        (LinearModel) The model instance.
    """
    assert polynomial_degree > 0, "Polynomial degree must be higher than 0!"
    if type(model) == str:
        model = create_model(model)
    data = Preprocessing.preprocess(train_dataset, polynomial_degree=polynomial_degree)
    logging.debug("Fitting training set to model")
    model.fit(data, train_dataset.target)
    return model
