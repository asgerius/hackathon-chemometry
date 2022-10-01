import numpy as np
from scipy.stats import linregress

from src.data import Data, load_from_pickle


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

if __name__ == "__main__":
    data = load_from_pickle()
    lr = combined_linear(data)
