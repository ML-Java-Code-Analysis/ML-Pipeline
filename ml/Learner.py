#!/usr/bin/python
# coding=utf-8
from sklearn import linear_model

MODEL_TYPE_LINREG = 'LINEAR_REGRESSION'


def train_model(model_type, train_dataset):
    model = create_model(model_type)
    model.fit(train_dataset.data, train_dataset.target)
    return model


def create_model(model_type):
    if model_type == MODEL_TYPE_LINREG:
        return linear_model.LinearRegression()
    else:
        raise ValueError("The model type %s is not supported." % model_type)
