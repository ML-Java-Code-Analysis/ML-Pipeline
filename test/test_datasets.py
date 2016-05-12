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
