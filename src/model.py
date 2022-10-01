from __future__ import annotations

import abc
from cProfile import label
import pickle

import numpy as np
from sklearn.cross_decomposition import PLSRegression 

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

        labels = data.labels.ravel()
        features = data.features.reshape(-1, 700)

        self.pls.fit(features, labels)

    def predict(self, data: Data) -> np.ndarray:
        
        features = data.features.reshape(-1,700)
        
        prediction = self.predict(features).reshape(())
















