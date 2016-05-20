#!/usr/bin/python
# coding=utf-8
import numpy as np
from scipy.sparse.csc import csc_matrix
from scipy.sparse.csr import csr_matrix


def log_transform(x, base='n'):
    if type(x) in (csc_matrix, csr_matrix):
        x.data = log_transform(x.data, base)
        return x

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
    x[x < 0] = np.finfo(np.float64).min  # Remove -inf from array, replace with best possible approximation.
    return x


def exp_transform(x, base='n'):
    if type(x) in (csc_matrix, csr_matrix):
        x.data = exp_transform(x.data, base)
        return x

    if str(base) == 'n':
        return np.exp(x)
    elif str(base).isnumeric():
        base = int(base)
        return np.power(base, x)
    else:
        raise ValueError("'%s' is not a valid base for log transform!")
