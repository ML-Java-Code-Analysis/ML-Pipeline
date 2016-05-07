#!/usr/bin/python
# coding=utf-8
import logging
import numpy as np
from sklearn import linear_model
from sklearn import svm
from sklearn.grid_search import GridSearchCV

from ml import Preprocessing

MODEL_TYPE_LINREG = 'LINEAR_REGRESSION'
MODEL_TYPE_RIDREG = 'RIDGE_REGRESSION'
MODEL_TYPE_SVR = 'SVR'


def create_model(model_type, normalize=False, cross_validation=False, cv=None, alpha=None, alpha_range=None, C=None,
                 C_range=None, kernel=None):
    """ Creates a new model of the specified type.

    Args:
        model_type (str): The type of model to create. Use one of the MODEL_TYPE_X constants.
        normalize (bool): If normalization is to be used.
        cross_validation (bool): If cross validation is to be applied, if applicable to the model type.
        alpha (float): The regularization parameter. Will only be used if applicable to the model type.
        kernel (str): The kernel to use, if applicable to the model type.

    Returns:
        (LinearModel) The model instance.
    """
    model_type = model_type.upper()
    logging.debug("Creating model with type %s" % model_type)
    if model_type == MODEL_TYPE_LINREG:
        return create_linear_regression_model(normalize)
    elif model_type == MODEL_TYPE_RIDREG:
        if cross_validation:
            return create_ridge_cv_model(alpha_range, normalize)
        else:
            return create_ridge_model(alpha, normalize)
    elif model_type == MODEL_TYPE_SVR:
        if cross_validation:
            return create_svr_cv_model(C_range, kernel)
        else:
            return create_svr_model(C, kernel)
    else:
        raise ValueError("The model type %s is not supported." % model_type)


def create_svr_cv_model(C_range=None, kernel=None):
    return GridSearchCV(
        estimator=create_svr_model(0, kernel),
        param_grid=dict(C=C_range),
        n_jobs=-1)


def create_svr_model(C=None, kernel=None):
    return svm.SVR(
        kernel=kernel,
        C=C,
        cache_size=8000,
    )


def create_ridge_model(alpha=None, normalize=None):
    return linear_model.Ridge(
        alpha=alpha,
        fit_intercept=True,
        normalize=normalize,
        copy_X=True,
    )


def create_ridge_cv_model(alpha_range=None, normalize=None, cv=5):
    return linear_model.RidgeCV(
        alphas=alpha_range,
        cv=cv,
        normalize=normalize,
    )


def create_linear_regression_model(normalize=None):
    return linear_model.LinearRegression(
        fit_intercept=True,
        normalize=normalize,
        copy_X=True,
    )


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
