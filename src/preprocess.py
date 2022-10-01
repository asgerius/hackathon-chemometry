from __future__ import annotations

import numpy as np
from scipy.stats import linregress

from src.data import Data, load_from_pickle

def standardize(data: Data) -> tuple[np.ndarray, np.ndarray]:
    mean = data.features.reshape((-1, 700)).mean(axis=0)
    std = data.features.reshape((-1, 700)).std(axis=0) + 1e-6
    return mean, std

def apply_standardize(data: Data, mean: np.ndarray, std: np.ndarray):
    """ INPLACE """
    data.features = data.features - mean
    data.features = data.features / std

def combined_linear(data: Data):
    """ Performs linear regression on all data and subtracts the line from each data point.
    The linear regression result is a model parameter and so is returned. """
    features = data.features.reshape((-1, 700))
    lr = linregress(
        np.vstack([data.nm] * len(features)).ravel(),
        features.ravel(),
    )
    return lr

def apply_combined_linear(data: Data, lr):
    """ INPLACE """
    data.features = data.features - (lr.slope * data.nm + lr.intercept)
