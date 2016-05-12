#!/usr/bin/python
# coding=utf-8
import logging

from sklearn import linear_model
from sklearn import svm
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing.data import PolynomialFeatures, StandardScaler

MODEL_TYPE_LINREG = 'LINEAR_REGRESSION'
MODEL_TYPE_RIDREG = 'RIDGE_REGRESSION'
MODEL_TYPE_SVR = 'SVR'


# noinspection PyPep8Naming
def create_model(model_type, feature_scaling=False, polynomial_degree=1, cross_validation=False, alpha=1.0,
                 alpha_range=None, C=None, C_range=None, kernel=None):
    """ Creates a new model of the specified type.

    Args:
        model_type (str): The type of model to create. Use one of the MODEL_TYPE_X constants.
        feature_scaling (bool): If feature scaling is to be used.
        polynomial_degree (int): If higher than 1, polynomial feature transformation will be applied.
        cross_validation (bool): If cross validation is to be applied, if applicable to the model type.
        alpha (float): The regularization parameter. Will only be used if applicable to the model type.
        alpha_range (list[float]): A range of regularization parameters. Will only be used with cross validation.
        C: The regularization parameter für SVR. Will only be used if applicable to the model type.
        C_range: A range of regularization parameters for SVR. Will only be used with cross validation.
        kernel (str): The kernel to use, if applicable to the model type.

    Returns:
        (sklearn.pipeline.Pipeline) The estimator model.
    """
    assert polynomial_degree > 0, "Polynomial degree must be higher than 0!"
    model_type = model_type.upper()
    logging.debug("Creating model with type %s" % model_type)
    if model_type == MODEL_TYPE_LINREG:
        model = create_linear_regression_model()
    elif model_type == MODEL_TYPE_RIDREG:
        if cross_validation:
            model = create_ridge_cv_model(alpha_range)
        else:
            model = create_ridge_model(alpha)
    elif model_type == MODEL_TYPE_SVR:
        if cross_validation:
            model = create_svr_cv_model(C_range, kernel)
        else:
            model = create_svr_model(C, kernel)
    else:
        raise ValueError("The model type %s is not supported." % model_type)

    steps = []
    if polynomial_degree > 1:
        steps.append(("poly", PolynomialFeatures(degree=polynomial_degree)))
    if feature_scaling:
        steps.append(("scale", StandardScaler()))
    steps.append((model_type, model))

    return Pipeline(steps)


# noinspection PyPep8Naming
def create_svr_cv_model(C_range=None, kernel=None):
    return GridSearchCV(
        estimator=create_svr_model(0, kernel),
        param_grid=dict(C=C_range),
        n_jobs=-1)


# noinspection PyPep8Naming
def create_svr_model(C=None, kernel=None):
    return svm.SVR(
        kernel=kernel,
        C=C,
        cache_size=8000,
    )


def create_ridge_model(alpha=1.0):
    return linear_model.Ridge(
        alpha=alpha,
        fit_intercept=True,
        copy_X=True,
    )


def create_ridge_cv_model(alpha_range=None, cv=5):
    return linear_model.RidgeCV(
        alphas=alpha_range,
        cv=cv,
    )


def create_linear_regression_model():
    return linear_model.LinearRegression(
        fit_intercept=True,
        copy_X=True,
    )


def train_model(model, train_dataset):
    """ Trains a model.

    Args:
        model (sklearn.pipeline.Pipeline): The model or pipeline to train.
        train_dataset (Dataset): The dataset to train the model with.

    Returns:
        (sklearn.pipeline.Pipeline) The trained estimator model.
    """
    logging.debug("Fitting training set to model")
    model.fit(train_dataset.data, train_dataset.target)
    return model
