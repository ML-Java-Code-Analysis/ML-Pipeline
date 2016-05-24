#!/usr/bin/python
# coding=utf-8

import warnings

import numpy as np
from numpy.lib.stride_tricks import as_strided
from scipy import sparse
from sklearn.preprocessing.data import StandardScaler, _handle_zeros_in_scale
from sklearn.utils import check_array
from sklearn.utils.extmath import _incremental_mean_and_var
from sklearn.utils.sparsefuncs import (inplace_column_scale,
                                       mean_variance_axis, incr_mean_variance_axis)
from sklearn.utils.validation import check_is_fitted, FLOAT_DTYPES


def sum(X, v):
    rows, cols = X.shape
    row_start_stop = as_strided(X.indptr, shape=(rows, 2),
                                strides=2 * X.indptr.strides)
    for row, (start, stop) in enumerate(row_start_stop):
        data = X.data[start:stop]
        data -= v[row]


class SparseScaler(StandardScaler):
    def __init__(self, copy=True, with_mean=True, with_std=True):
        super(SparseScaler, self).__init__(copy=copy, with_mean=with_mean, with_std=with_std)

    def partial_fit(self, X, y=None):
        """Online computation of mean and std on X for later scaling.
        All of X is processed as a single batch. This is intended for cases
        when `fit` is not feasible due to very large number of `n_samples`
        or because X is read from a continuous stream.

        The algorithm for incremental mean and std is given in Equation 1.5a,b
        in Chan, Tony F., Gene H. Golub, and Randall J. LeVeque. "Algorithms
        for computing the sample variance: Analysis and recommendations."
        The American Statistician 37.3 (1983): 242-247:

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape [n_samples, n_features]
            The data used to compute the mean and standard deviation
            used for later scaling along the features axis.

        y: Passthrough for ``Pipeline`` compatibility.
        """
        X = check_array(X, accept_sparse=('csr', 'csc'), copy=self.copy,
                        ensure_2d=False, warn_on_dtype=True,
                        estimator=self, dtype=FLOAT_DTYPES)

        # Even in the case of `with_mean=False`, we update the mean anyway
        # This is needed for the incremental computation of the var
        # See incr_mean_variance_axis and _incremental_mean_variance_axis

        if not sparse.issparse(X):
            return super(SparseScaler, self).partial_fit(X)

        if self.with_std:
            # First pass
            if not hasattr(self, 'n_samples_seen_'):
                self.mean_, self.var_ = mean_variance_axis(X, axis=0)
                n = X.shape[0]
                self.n_samples_seen_ = n
            # Next passes
            else:
                self.mean_, self.var_, self.n_samples_seen_ = \
                    incr_mean_variance_axis(X, axis=0,
                                            last_mean=self.mean_,
                                            last_var=self.var_,
                                            last_n=self.n_samples_seen_)

        if self.with_std:
            self.scale_ = _handle_zeros_in_scale(np.sqrt(self.var_))
        else:
            self.scale_ = None

        return self

    def transform(self, X, y=None, copy=None):
        """Perform standardization by centering and scaling

        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data used to scale along the features axis.
        """
        check_is_fitted(self, 'scale_')

        copy = copy if copy is not None else self.copy
        X = check_array(X, accept_sparse='csr', copy=copy,
                        ensure_2d=False, warn_on_dtype=True,
                        estimator=self, dtype=FLOAT_DTYPES)

        if not sparse.issparse(X):
            return super(SparseScaler, self).transform(X, y=y, copy=copy)

        if self.with_mean:
            # X -= self.mean_
            sum(X, self.mean_ * -1)
        if self.with_std:
            X /= self.scale_
        return X

    def inverse_transform(self, X, copy=None):
        """Scale back the data to the original representation

        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data used to scale along the features axis.
        """
        check_is_fitted(self, 'scale_')

        copy = copy if copy is not None else self.copy
        if not sparse.issparse(X):
            super(SparseScaler, self).inverse_transform(X, copy=copy)

        if self.with_mean:
            raise ValueError(
                "Cannot uncenter sparse matrices: pass `with_mean=False` "
                "instead See docstring for motivation and alternatives.")
        if not sparse.isspmatrix_csr(X):
            X = X.tocsr()
            copy = False
        if copy:
            X = X.copy()

        if self.mean_ is not None:
            sum(X, self.mean_)
        if self.scale_ is not None:
            inplace_column_scale(X, self.scale_)
        return X
