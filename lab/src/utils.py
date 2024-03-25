import scipy.stats
import warnings
import functools
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from copy import deepcopy
import abc
from dataclasses import dataclass, field


def deprecated(func):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used."""
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.simplefilter('always', DeprecationWarning)  # turn off filter
        warnings.warn("Call to deprecated function {}.".format(func.__name__),
                      category=DeprecationWarning,
                      stacklevel=2)
        warnings.simplefilter('default', DeprecationWarning)  # reset filter
        return func(*args, **kwargs)
    return new_func


def baseline_als(y, **kwargs):
    lam = kwargs.get("lam")
    p = kwargs.get("p")
    niter = kwargs.get("niter")

    L = len(y)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L-2))
    w = np.ones(L)
    for i in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = spsolve(Z, w*y)
        w = p * (y > z) + (1-p) * (y < z)
    # return deepcopy(z)
    return deepcopy(z)


def baseline_remove(df, **kwargs):
    """Remove baseline"""

    for column in df.columns:
        bsline = baseline_als(df[column], **kwargs)
        df[column] = bsline
    return df


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h


class Stats(abc.ABC):
    """Class to represent the statistics of the data"""
    data: DataFrame = field(init=False)
    mean: np.array = np.array([])
    std: np.array = np.array([])
    min: np.array = np.array([])
    max: np.array = np.array([])
    median: np.array = np.array([])
    confidence_value: float = 0.95
    confidence_interval: np.array = np.array([])
# """Class to represent the statistics of the data"""
#    data = DataFrame = field(init=False)
#    mean: np.array = np.array([])
#    std: np.array = np.array([])
#    min: np.array = np.array([])
#    max: np.array = np.array([])
#    median: np.array = np.array([])
#    confidence_value: float = 0.95
#    confidence_interval: np.array = np.array([][][])
