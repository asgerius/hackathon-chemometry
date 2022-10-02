from __future__ import annotations
from sys import stderr
from tkinter import W

import numpy as np
from pyparsing import with_attribute
from scipy.stats import linregress

from src.data import Data, load_from_pickle

def standardize(data: Data) -> tuple[np.ndarray, np.ndarray]:
    mean = data.features.reshape((-1, data.num_features)).mean(axis=0)
    std = data.features.reshape((-1, data.num_features)).std(axis=0) + 1e-6
    return mean, std

def apply_standardize(data: Data, mean: np.ndarray, std: np.ndarray):
    """ INPLACE """
    data.features = data.features - mean
    data.features = data.features / std

def combined_linear(data: Data):
    """ Performs linear regression on all data and subtracts the line from each data point.
    The linear regression result is a model parameter and so is returned. """
    features = data.features.reshape((-1, data.num_features))
    lr = linregress(
        np.vstack([data.nm] * len(features)).ravel(),
        features.ravel(),
    )
    return lr

def get_logarithm(data: Data):
    flatfeatures = data.features.reshape(-1, data.num_features)
    flatfeatures = np.log(flatfeatures)
    return flatfeatures.reshape(data.features.shape)

def get_derivatives(data: Data):
    flatfeatures = data.features.reshape(-1, data.num_features)
    gradx = np.gradient(flatfeatures,axis=0)
    return gradx.reshape(data.features.shape)

def apply_derivatives(data: np.ndarray, derivatives: np.ndarray):
    return np.concatenate((data,derivatives), axis=-1)

def apply_logs(data: np.ndarray, logs: np.ndarray):
    return np.concatenate((data,logs), axis=-1)

def apply_combined_linear(data: Data, lr):
    """ INPLACE """
    data.features = data.features - (lr.slope * data.nm + lr.intercept)

def get_within(data: Data) -> tuple[np.ndarray, np.ndarray]:
    means = data.features.mean(axis=(1, 2))
    std = data.features.std(axis=(1, 2))
    return means, std

def apply_within(data: Data, within_mean: np.ndarray, within_std: np.ndarray):
    """ INPLACE """
    features = data.features
    wmean = np.stack(data.labels.shape[1] * [within_mean], axis=1)
    wmean = np.stack(data.labels.shape[2] * [wmean], axis=2)
    features = np.concatenate([features, wmean], axis=-1)

    # wstd = np.stack(data.labels.shape[1] * [within_std], axis=1)
    # wstd = np.stack(data.labels.shape[2] * [wstd], axis=2)
    # data.features = np.concatenate([features, wstd], axis=-1)

if __name__ == "__main__":

    data = load_from_pickle()
    lr = combined_linear(data)
    apply_combined_linear(data, lr)
    print(data.features.shape)

    m, s = get_within(data)
    apply_within(data, m, s)
    print(data.features.shape)
