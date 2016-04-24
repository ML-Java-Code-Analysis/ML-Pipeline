#!/usr/bin/python
# coding=utf-8
import logging

from sklearn import linear_model

MODEL_TYPE_LINREG = 'LINEAR_REGRESSION'


def train_model(model, train_dataset):
    if type(model) == str:
        model = create_model(model)
    logging.info("Fitting training set to model")
    model.fit(train_dataset.data, train_dataset.target)
    return model


def create_model(model_type):
    model_type = model_type.upper()
    logging.debug("Creating model with type %s" % model_type)
    if model_type == MODEL_TYPE_LINREG:
        return linear_model.LinearRegression()
    else:
        raise ValueError("The model type %s is not supported." % model_type)
