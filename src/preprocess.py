from __future__ import annotations

import numpy as np
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

if __name__ == "__main__":

    data = load_from_pickle()
    lr = combined_linear(data)
    apply_combined_linear(data, lr)
    logs=get_logarithm(data)
    ders=get_derivatives(data)
    print(np.shape(data.features))
    print(np.shape(ders))
    data.features=apply_derivatives(data.features,ders)
    print(np.shape(data.features))
    print(logs.shape)
    data.features = apply_logs(data.features,logs)
    print(np.shape(data.features))