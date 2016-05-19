#!/usr/bin/python
# coding=utf-8
import numpy as np


def log_transform(x, base='n'):
    if str(base) == 'n':
        x = np.log(x)
    elif str(base).isnumeric():
        base = int(base)
        if base == 2:
            x = np.log2(x)
        elif base == 10:
            x = np.log10(x)
        else:
            x = np.log(x) / np.log(base)
    else:
        raise ValueError("'%s' is not a valid base for log transform!")
    x[x<0] = 0 # Remove -inf from array
    return x


def exp_transform(x, base='n'):
    if str(base) == 'n':
        return np.exp(x)
    elif str(base).isnumeric():
        base = int(base)
        return np.power(base, x)
    else:
        raise ValueError("'%s' is not a valid base for log transform!")
