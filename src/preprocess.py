from __future__ import annotations
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

def get_within_mean(data: Data):
    flatfeatures = data.features.reshape(-1,700)
    within_mean = np.zeros_like(flatfeatures)
    sum = 0
    for sample in range(int(flatfeatures.shape[0]/data.features.shape[2])):
        #print(flatfeatures[sample*(data.features.shape[2]):sample*(data.features.shape[2])+31].shape)
        mean = flatfeatures[sample*(data.features.shape[2]):sample*(data.features.shape[2])+31].std(axis=0)
        #print(mean.shape)
        #print(mean)
        for scan in range (32):
            within_mean[scan+sample*data.features.shape[2]] = mean
    #print(within_mean)
    return within_mean.reshape(data.features.shape)

def get_within_sd(data: Data):
    flatfeatures = data.features.reshape(-1,700)
    within_sd = np.zeros_like(flatfeatures)
    sum = 0
    for sample in range(int(flatfeatures.shape[0]/data.features.shape[2])):
        sd = flatfeatures[sample*(data.features.shape[2]):sample*(data.features.shape[2])+31].mean(axis=0)+ 1e-6
        for scan in range (32):
            within_sd[scan+sample*data.features.shape[2]] = sd
    return within_sd.reshape(data.features.shape)

def apply_within_mean(data: Data, within_mean: Data):
    return np.concatenate((data.features,within_mean), axis=-1)

def apply_within_sd(data: Data, within_sd: Data):
    return np.concatenate((data.features,within_sd), axis=-1)

if __name__ == "__main__":

    data = load_from_pickle()
    lr = combined_linear(data)
    apply_combined_linear(data, lr)
    print(data.features.shape) 
    wmean = get_within_mean(data)
    print(wmean.shape)
    wsd = get_within_sd(data)
    print(wsd.shape)
    data.features = apply_within_mean(data,wmean)
    print(data.features.shape) 
    data.features = apply_within_sd(data,wsd)
    print(data.features.shape) 