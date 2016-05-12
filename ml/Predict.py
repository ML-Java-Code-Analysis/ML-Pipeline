#!/usr/bin/python
# coding=utf-8
import random

import numpy as np


def predict_mean(training_dataset, length):
    mean = np.mean(training_dataset.target)
    return np.array([mean] * length)


def predict_median(training_dataset, length):
    median = np.median(training_dataset.target)
    return np.array([median] * length)


def predict_weighted_random(training_dataset, length):
    weighted_values = {}
    for target_value in training_dataset.target:

        if type(target_value) == np.ndarray:
            target_value = target_value[0]
        weighted_values[target_value] = weighted_values.get(target_value, 0) + 1
    prediction = np.zeros(length)
    for i in range(length):
        prediction[i] = weighted_choice(weighted_values)
    return prediction


def weighted_choice(choices):
    """

    Source: http://stackoverflow.com/questions/3679694/a-weighted-version-of-random-choice

    Args:
        choices (:

    Returns:

    """
    total = sum(choices.values())
    r = random.uniform(0, total)
    upto = 0
    for c, w in choices.items():
        if upto + w >= r:
            return c
        upto += w
    assert False, "Shouldn't get here"


def predict_with_model(dataset, model):
    return model.predict(dataset.data)
