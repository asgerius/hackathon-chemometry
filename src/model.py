from __future__ import annotations

import abc
import pickle
from pyexpat import features

import numpy as np
import pelutils.ds.distributions as dists
from sklearn.cross_decomposition import PLSRegression
from scipy.stats import mode
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge, SGDClassifier

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

    def __init__(self, n_components = 80):
        self.pls = PLSRegression(n_components=n_components, max_iter= 5000)

    def fit(self, data: Data):
        features = data.features.reshape((-1, data.features.shape[-1]))
        labels = data.one_hot_labels().reshape(-1,3)
        self.pls.fit(features, labels)

    def predict(self, data: Data) -> np.ndarray:
        features = data.features.reshape((-1, data.features.shape[-1]))
        preds = self.pls.predict(features)
        preds = preds.argmax(axis=1) + 1
        preds = preds.reshape(data.labels.shape)
        preds = mode(preds, axis=2, keepdims=True).mode
        return np.squeeze(preds)

    def __str__(self) -> str:
        return "PartialLeastSquares(n_components =%d)"% self.pls.n_components

class Mixed(Model):
    def __init__(self, n_components = 50, alpha = 0.01, n_estimators=500):
        self.pls = PLSRegression(n_components=n_components, max_iter= 5000)
        self.ridge = Ridge(alpha=alpha, max_iter= 5000)
        # self.forest = RandomForestClassifier(n_estimators=n_estimators)

    def fit(self, data: Data):
        features = data.features.reshape((-1, data.features.shape[-1]))
        labels = data.one_hot_labels().reshape(-1,3)
        self.pls.fit(features, labels)
        self.ridge.fit(features, labels)
        # self.forest.fit(features, labels)

    def predict(self, data: Data) -> np.ndarray:
        features = data.features.reshape((-1, data.features.shape[-1]))
        preds_pls = self.pls.predict(features).argmax(axis=1) + 1
        preds_ridge = self.ridge.predict(features).argmax(axis=1) + 1
        # preds_forest = self.forest.predict(features).argmax(axis=1) + 1
        preds = np.concatenate((preds_pls,preds_ridge), axis= -1)
        # preds = np.concatenate((preds,preds_forest), axis= -1)

        preds = preds.reshape(data.labels.shape)
        preds = mode(preds, axis=2, keepdims=True).mode
        return np.squeeze(preds)

    def predict(self, data: Data) -> np.ndarray:
        features = data.features.reshape((-1, data.features.shape[-1]))
        preds = self.ridge.predict(features).argmax(axis=1) + 1
        preds = preds.reshape(data.labels.shape)
        preds = mode(preds, axis=-1).mode
        return np.squeeze(preds)

    def __str__(self) -> str:
        return "PartialLeastSquares(n_components =%d)"% self.pls.n_components
        

class SDG(Model):

    def __init__(self, alpha = 0.001, max_iter=100):
        self.SGD = SGDClassifier(alpha = alpha, max_iter=max_iter)

    def fit(self, data: Data):
        features = data.features.reshape((-1, data.features.shape[-1]))
        # labels = data.one_hot_labels().reshape(-1,3)
        labels = data.labels.ravel()
        print(labels.shape)
        self.SGD.fit(features, labels)

    def predict(self, data: Data) -> np.ndarray:
        features = data.features.reshape((-1, data.features.shape[-1]))
        preds = self.SGD.predict(features)
        
        preds = preds.reshape(data.labels.shape)
        print(preds.shape)
        # preds = preds.argmax(axis=1) + 1
        preds = mode(preds, axis=2, keepdims=True).mode
        print(preds.shape)
        return 

    def __str__(self) -> str:
        return "SDG"


class RidgeRegression(Model):

    def __init__(self, alpha=0.001):
        self.ridge = Ridge(alpha=alpha, max_iter= 5000)

    def fit(self, data: Data):
        features = data.features.reshape((-1, data.features.shape[-1]))
        labels = data.one_hot_labels().reshape(-1, 3)
        self.ridge.fit(features, labels)

    def predict(self, data: Data) -> np.ndarray:
        features = data.features.reshape((-1, data.features.shape[-1]))
        preds = self.ridge.predict(features).argmax(axis=1) + 1
        preds = preds.reshape(data.labels.shape)
        preds = mode(preds, axis=-1).mode
        return np.squeeze(preds)

    def __str__(self) -> str:
        return "RidgeRegression(alpha=%s)" % self.ridge.alpha

class StatShit(Model):

    def fit(self, data: Data):
        self.dists = list()
        log_transform = np.log(data.features.reshape((-1, data.num_features)))
        labels = data.labels.ravel()

        for label in range(data.num_labels):
            mus = log_transform[labels==label+1].mean(axis=0)
            sigma2s = log_transform[labels==label+1].var(axis=0)
            self.dists.append([dists.lognorm(mu, sigma2) for mu, sigma2 in zip(mus, sigma2s)])

    def predict(self, data: Data) -> np.ndarray:
        features = data.features.reshape((-1, data.num_features))
        log_liks = np.zeros((*features.shape, data.num_labels))

        for i in range(data.num_labels):
            for j in range(data.num_features):
                log_liks[:, j, i] = log_liks[:, j, i] + np.log(self.dists[i][j].pdf(features[:, j]))

        log_liks = log_liks.reshape(*data.features.shape, 3)
        total_log_lik = log_liks.sum(axis=(1, 2, 3))
        preds = total_log_lik.argmax(axis=1) + 1
        preds = np.vstack([preds]*3).T

        return np.squeeze(preds)

    def __str__(self) -> str:
        return "Log likelihood (lognorm)"

class RandomForest(Model):

    def __init__(self):
        self.forest = RandomForestClassifier()

    def fit(self, data: Data):
        features = data.features.reshape((-1, data.features.shape[-1]))
        labels = data.one_hot_labels().reshape(-1, 3)
        self.forest.fit(features, labels)

    def predict(self, data: Data) -> np.ndarray:
        features = data.features.reshape((-1, data.features.shape[-1]))
        preds = self.forest.predict(features).argmax(axis=1) + 1
        preds = preds.reshape(data.labels.shape)
        preds = mode(preds, axis=-1).mode
        return np.squeeze(preds)

    def __str__(self) -> str:
        return "RandomForest"
