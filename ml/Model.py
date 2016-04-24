#!/usr/bin/python
# coding=utf-8
import logging

from sklearn import linear_model
from sklearn.metrics import explained_variance_score, mean_squared_error, mean_absolute_error, r2_score

MODEL_TYPE_LINREG = 'LINEAR_REGRESSION'


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
            normalize=normalize
        )
    else:
        raise ValueError("The model type %s is not supported." % model_type)


def train_model(model, train_dataset):
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
    logging.debug("Fitting training set to model")
    model.fit(train_dataset.data, train_dataset.target)
    return model


def test_model(model, test_dataset):
    """ Test the trained model with the given dataset.

    Args:
        model (LinearModel): A trained Model instance.
        test_dataset (Dataset): The test dataset, including test input as well as target ground truth.
    """
    # logging.debug("Testing model with %s dataset" % test_dataset.size)
    predicted = model.predict(test_dataset.data)
    target = test_dataset.target

    evs = explained_variance_score(target, predicted)
    mse = mean_squared_error(target, predicted)
    mae = mean_absolute_error(target, predicted)
    r2s = r2_score(target, predicted)

    print("PREDICTED:\n" + str(predicted))
    print("TARGET:\n" + str(target))
    print("ABS DIFF:\n" + str(abs(predicted - target)))
    print("Explained variance score:\t%f\t(Best is 1.0, lower is worse)" % evs)
    print("Mean squared error:\t\t\t%f\t(Best is 0.0, higher is worse)" % mse)
    print("Mean absolute error:\t\t%f\t(Best is 0.0, higher is worse)" % mae)
    print("R2 Score:\t\t\t\t\t%f\t(Best is 1.0, lower is worse)" % r2s)
