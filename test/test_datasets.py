#!/usr/bin/python
# coding=utf-8
import datetime

import numpy as np

from ml.Dataset import Dataset


def get_simple_linear_datasets():
    return get_simple_linear_train_dataset(), get_simple_linear_test_dataset()


def get_simple_linear_train_dataset():
    dataset = Dataset(
        5,
        10,
        ["F1, F2, F3, F4, F5"],
        "sixmonth_bugs",
        datetime.date(2015, 1, 1),
        datetime.date(2015, 1, 31),
        False,
        "SimpleLinearTrain"
    )

    dataset.data = np.array([
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [10.0, 0.0, 0.0, 0.0, -5.0],
        [10.0, 0.0, 0.0, 0.0, -5.0],
        [10.0, 0.0, 0.0, 0.0, -5.0],
        [20.0, 0.0, 0.0, 0.0, -10.0],
        [50.0, 0.0, 0.0, 0.0, -25.0],
    ])

    dataset.target = np.array([
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
        1.0,
        2.0,
        5.0,
    ])

    return dataset


def get_simple_linear_test_dataset():
    dataset = Dataset(
        5,
        10,
        ["F1, F2, F3, F4, F5"],
        "sixmonth_bugs",
        datetime.date(2015, 2, 1),
        datetime.date(2015, 2, 28),
        False,
        "SimpleLinearTest"
    )

    dataset.data = np.array([
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [10.0, 0.0, 0.0, 0.0, -5.0],
        [20.0, 0.0, 0.0, 0.0, -10.0],
        [30.0, 0.0, 0.0, 0.0, -15.0],
        [40.0, 0.0, 0.0, 0.0, -20.0],
        [50.0, 0.0, 0.0, 0.0, -25.0],
        [60.0, 0.0, 0.0, 0.0, -30.0],
        [70.0, 0.0, 0.0, 0.0, -35.0],
        [80.0, 0.0, 0.0, 0.0, -40.0],
        [90.0, 0.0, 0.0, 0.0, -45.0],
    ])

    dataset.target = np.array([
        0.0,
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
    ])

    return dataset


def get_simple_polynomial_datasets():
    # return get_simple_polynomial_train_dataset(), get_simple_polynomial_test_dataset()
    return get_simple_polynomial_dataset("Train", sample_size=100), get_simple_polynomial_dataset("Test")


def get_simple_polynomial_dataset(label, sample_size=10):
    dataset = Dataset(
        5,
        10,
        ["F1, F2, F3, F4, F5"],
        "sixmonth_bugs",
        datetime.date(2015, 1, 1),
        datetime.date(2015, 1, 31),
        False,
        label
    )

    dataset.data = np.random.rand(sample_size, 5)

    dataset.target = np.zeros((sample_size, 1))
    for i, row in enumerate(dataset.data):
        dataset.target[i] = 8 + 3 * row[0] + row[1] + row[2] ** 2 + row[2] * row[4]

    return dataset


def get_simple_polynomial_train_dataset():
    dataset = Dataset(
        5,
        10,
        ["F1, F2, F3, F4, F5"],
        "sixmonth_bugs",
        datetime.date(2015, 1, 1),
        datetime.date(2015, 1, 31),
        False,
        "SimpleLinearTrain"
    )

    dataset.data = np.array([
        [0.0, 1.0, 5.0, 2.0, 0.0],
        [0.0, 1.0, 5.0, 3.0, 0.0],
        [0.0, 1.0, 5.0, 1.0, 0.0],
        [0.0, 1.0, 5.0, 6.0, 0.0],
        [0.0, 1.0, 5.0, 2.0, 0.0],
        [10.0, 1.0, 25.0, 7.0, -5.0],
        [10.0, 1.0, 25.0, 2.0, -5.0],
        [10.0, 1.0, 25.0, 3.0, -5.0],
        [20.0, 1.0, 6.0, 1.0, -10.0],
        [50.0, 1.0, 6.0, 7.0, -25.0],
    ])

    dataset.target = np.zeros((10, 1))

    for i, row in enumerate(dataset.data):
        dataset.target[i] = 3 + row[0] + row[2] ** 2 + row[2] * row[4]

    return dataset


def get_simple_polynomial_test_dataset():
    dataset = Dataset(
        5,
        10,
        ["F1, F2, F3, F4, F5"],
        "sixmonth_bugs",
        datetime.date(2015, 1, 1),
        datetime.date(2015, 1, 31),
        False,
        "SimpleLinearTrain"
    )

    dataset.data = np.array([
        [10.0, 1.0, 5.0, 5.0, 0.0],
        [20.0, 1.0, 5.0, 3.0, 0.0],
        [30.0, 1.0, 5.0, 2.0, 1.0],
        [40.0, 1.0, 255.0, 2.0, 0.0],
        [50.0, 1.0, 5.0, 3.0, 0.0],
        [60.0, 1.0, 2.0, 67.0, -5.0],
        [70.0, 1.0, 23.0, 8.0, -1.0],
        [80.0, 1.0, 21.0, 3.0, -5.0],
        [90.0, 1.0, 6.0, 2.0, -20.0],
        [100.0, 1.0, 5.0, 6.0, -25.0],
    ])

    dataset.target = np.zeros((10, 1))

    for i, row in enumerate(dataset.data):
        dataset.target[i] = 3 + row[0] + row[2] ** 2 + row[2] * row[4]

    return dataset
