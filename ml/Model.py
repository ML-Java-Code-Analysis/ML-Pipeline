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

KERNEL_LINEAR = 'linear'
KERNEL_POLYNOMIAL = 'poly'
KERNEL_RBF = 'rbf'
KERNEL_SIGMOID = 'sigmoid'


# noinspection PyPep8Naming
def create_model(model_type, feature_scaling=False, polynomial_degree=1, cross_validation=False, alpha=1.0, C=None,
                 kernel=None, svr_epsilon=None, svr_degree=None, svr_gamma=None, svr_coef0=None, sparse=False, ):
    """ Creates a new model of the specified type.

    Args:
        model_type (str): The type of model to create. Use one of the MODEL_TYPE_X constants.
        feature_scaling (bool): If feature scaling is to be used.
        polynomial_degree (int): If higher than 1, polynomial feature transformation will be applied.
        cross_validation (bool): If cross validation is to be applied, if applicable to the model type.
        alpha (float): The regularization parameter. Will only be used if applicable to the model type.
        C: The regularization parameter für SVR. Will only be used if applicable to the model type.
        kernel (str): The kernel to use, if applicable to the model type.
        sparse (bool): If a sparse feature matrix is used.
        svr_epsilon (float): Epsilon parameter for SVR. Specifies the epsilon tube. (see sklearn for more info)
        svr_degree (int): Polynomial degree parameter for the SVR kernel 'poly'
        svr_gamma (float): Kernel coefficient for SVR kernels 'rbf', 'poly' and 'sigmoid'
        svr_coef0 (float): Independent term (or bias) for SVR kernels 'poly' and 'sigmoid'

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
            model = create_ridge_cv_model(alpha)
        else:
            model = create_ridge_model(alpha)
    elif model_type == MODEL_TYPE_SVR:
        if cross_validation:
            model = create_svr_cv_model(C, kernel, svr_epsilon, svr_degree, svr_gamma, svr_coef0)
        else:
            model = create_svr_model(C, kernel, svr_epsilon, svr_degree, svr_gamma, svr_coef0)
    else:
        raise ValueError("The model type %s is not supported." % model_type)

    steps = []
    if polynomial_degree > 1:
        if not sparse:
            steps.append(("poly", PolynomialFeatures(degree=polynomial_degree)))
        else:
            logging.warning("Polynomial Features for sparse matrices are not supported!")
    if feature_scaling:
        if not sparse:
            steps.append(("scale", StandardScaler(with_mean=not sparse)))
        else:
            logging.warning("Sparse matrices cannot be scaled with mean. Only Std scaling will be applied.")
    steps.append((model_type, model))

    return Pipeline(steps)


def _to_list(x):
    if x is not None and type(x) != list:
        return [x]
    return x


# noinspection PyPep8Naming
def create_svr_cv_model(C=None, kernel='linear', epsilon=None, degree=None, gamma=None, coef0=None):
    param_grid = []
    if type(kernel) != list:
        kernel = [kernel]

    for kernel_ in _to_list(kernel):
        param_dict = {'kernel': [kernel_]}
        if C is not None:
            param_dict['C'] = _to_list(C)
        if epsilon is not None:
            param_dict['epsilon'] = _to_list(epsilon)

        if kernel_ == KERNEL_POLYNOMIAL:
            if degree is not None:
                param_dict['degree'] = _to_list(degree)
            if coef0 is not None:
                param_dict['coef0'] = _to_list(coef0)
        elif kernel_ == KERNEL_RBF:
            if gamma is not None:
                param_dict['gamma'] = _to_list(gamma)
        elif kernel_ == KERNEL_SIGMOID:
            if gamma is not None:
                param_dict['gamma'] = _to_list(gamma)
            if coef0 is not None:
                param_dict['coef0'] = _to_list(coef0)
        param_grid.append(param_dict)

    return GridSearchCV(
        estimator=svm.SVR(),
        param_grid=param_grid,
        n_jobs=-1)


def _get_first_if_list(x):
    if x is not None and type(x) == list and len(x) > 0:
        return x[0]
    return x


# noinspection PyPep8Naming
def create_svr_model(C=None, kernel=None, epsilon=None, degree=None, gamma=None, coef0=None):
    return svm.SVR(
        kernel=_get_first_if_list(kernel),
        C=_get_first_if_list(C),
        cache_size=8000,
        degree=_get_first_if_list(degree),
        epsilon=_get_first_if_list(epsilon),
        gamma=_get_first_if_list(gamma),
        coef0=_get_first_if_list(coef0)
    )


def create_ridge_model(alpha=1.0):
    return linear_model.Ridge(
        alpha=_get_first_if_list(alpha),
        fit_intercept=True,
        copy_X=True,
    )


def create_ridge_cv_model(alpha=1.0, cv=5):
    return GridSearchCV(
        estimator=create_ridge_model(0),
        param_grid=dict(alpha=_to_list(alpha)),
        iid=False,
        cv=cv,
        n_jobs=-1)


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
