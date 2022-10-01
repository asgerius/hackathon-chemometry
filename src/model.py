from __future__ import annotations

import abc
from cProfile import label
import pickle

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from scipy.stats import mode
from sklearn.linear_model import Ridge

from data import Data




class Model(abc.ABC):

    @abc.abstractmethod
    def fit(self, data: Data):
        pass

    @abc.abstractmethod
    def predict(self, data: Data) -> np.ndarray:
        pass

    def save(self, path: str):
        path = f"{path}/model.pkl"
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> Model:
        path = f"{path}/model.pkl"
        with open(path, "rb") as f:
            return pickle.load(path)

    @abc.abstractmethod
    def __str__(self) -> str:
        pass

class Baseline(Model):

    def fit(self, data: Data):
        pass

    def predict(self, data: Data) -> np.ndarray:
        return np.ones((data.features.shape[0], 3), dtype=int)

    def __str__(self) -> str:
        return "Baseline(1)"

class PartialLeastSquares(Model):

    def fit(self, data: Data):
        self.pls = PLSRegression(n_components=3)

        features = data.features.reshape(-1, 700)
        labels = data.one_hot_labels().reshape(-1,3)
        self.pls.fit(features, labels)

    def predict(self, data: Data) -> np.ndarray:
        
        features = data.features.reshape(-1,700)
        preds = self.pls.predict(features)
        preds = preds.argmax(axis=1) + 1
        preds = preds.reshape(data.labels.shape)
        preds = mode(preds, axis=2, keepdims=True).mode
        return np.squeeze(preds)

    def __str__(self) -> str:
        return "PartialLeastSquares"

class RidgeRegression(Model):

    def __init__(self, alpha=0):
        self.ridge = Ridge(alpha=alpha)

    def fit(self, data: Data):
        features = data.features.reshape((-1, 700))
        labels = data.one_hot_labels().reshape(-1, 3)
        self.ridge.fit(features, labels)

    def predict(self, data: Data) -> np.ndarray:
        features = data.features.reshape((-1, 700))
        preds = self.ridge.predict(features).argmax(axis=1) + 1
        preds = preds.reshape(data.labels.shape)
        preds = mode(preds, axis=-1).mode
        return np.squeeze(preds)

    def __str__(self) -> str:
        return "RidgeRegression(alpha=%s)" % self.ridge.alpha
