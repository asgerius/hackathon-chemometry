from __future__ import annotations

import abc
import pickle
from dataclasses import dataclass

import numpy as np
import pelutils.ds.distributions as dists
from sklearn import neighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from pelutils import log
from sklearn.cross_decomposition import PLSRegression
from scipy.stats import mode
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge, SGDClassifier
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import svm
from tqdm import tqdm

from data import Data


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    def __init__(self, n_components = 50, alpha = 0.01):
        self.pls = PLSRegression(n_components=n_components, max_iter= 5000)
        self.ridge = Ridge(alpha=alpha, max_iter= 5000)

    def fit(self, data: Data):
        features = data.features.reshape((-1, data.features.shape[-1]))
        labels = data.one_hot_labels().reshape(-1,3)
        self.pls.fit(features, labels)
        self.ridge.fit(features, labels)

    def predict(self, data: Data) -> np.ndarray:
        features = data.features.reshape((-1, data.features.shape[-1]))
        preds_pls = self.pls.predict(features).argmax(axis=1) + 1
        preds_ridge = self.ridge.predict(features).argmax(axis=1) + 1
        preds = np.concatenate((preds_pls,preds_ridge), axis= -1)

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

class MixedBagging(Model):
    def __init__(self, n=50, n_components = 50, alpha = 0.001):
        self.n = n
        self.models = [
            [PLSRegression(n_components=n_components) for _ in range(n)],
            [Ridge(alpha=alpha) for _ in range(n)],
        ]
        self.num_models = len(self.models)

    def fit(self, data: Data):

        for i, d in tqdm(enumerate(data.bagging(self.n)), total=self.n):
            features = d.features.reshape((-1, d.features.shape[-1]))
            labels = d.one_hot_labels().reshape(-1, 3)

            for j in range(self.num_models):
                self.models[j][i].fit(features, labels)

    def predict(self, data: Data) -> np.ndarray:

        preds = [np.empty((self.n, *data.labels.shape, 3)) for _ in range(self.num_models)]
        for i in range(self.n):
            features = data.features.reshape((-1, data.num_features))
            for j in range(self.num_models):
                preds[j][i] = self.models[j][i].predict(features).reshape((*data.labels.shape, 3))

        preds = np.concatenate(preds, axis=0)
        preds = preds.transpose(1, 2, 3, 0, 4)
        lab_preds = preds.argmax(axis=-1) + 1
        lab_preds = lab_preds.reshape((*lab_preds.shape[:-2], -1))
        lab_preds = mode(lab_preds, axis=-1).mode
        return np.squeeze(lab_preds)


    # def predict(self, data: Data) -> np.ndarray:
    #     features = data.features.reshape((-1, data.features.shape[-1]))
    #     preds = self.ridge.predict(features).argmax(axis=1) + 1
    #     preds = preds.reshape(data.labels.shape)
    #     preds = mode(preds, axis=-1).mode
    #     return np.squeeze(preds)

    def __str__(self) -> str:
        return "Mixed Bagging"

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

class KNN(Model):
    def __init__(self,n_neighbors=8, weights="distance"):
        self.KNN = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)        
        
    def fit(self, data: Data):
        features = data.features.reshape((-1, data.features.shape[-1]))
        labels = data.one_hot_labels().reshape(-1, 3)
        print(labels.shape)
        print(features.shape)
        self.KNN.fit(features, labels)

    def predict(self, data: Data):
        features = data.features.reshape((-1, data.features.shape[-1]))
        preds = self.KNN.predict(features).argmax(axis=1) + 1
        preds = preds.reshape(data.labels.shape)
        preds = mode(preds, axis=-1).mode
        return np.squeeze(preds)
    
    def __str__(self) -> str:
        return "KNN"

class LDA(Model):
    def __init__(self,solver="svd"):
        self.LDA = LinearDiscriminantAnalysis(solver=solver)
        
    def fit(self, data: Data):
        features = data.features.reshape((-1, data.features.shape[-1]))
        #labels = data.one_hot_labels().reshape(-1, 3)
        labels = data.labels.ravel()
        print(labels.shape)
        print(features.shape)
        self.LDA.fit(features, labels)

    def predict(self, data: Data):
        features = data.features.reshape((-1, data.features.shape[-1]))
        preds = self.LDA.predict(features)
        preds = preds.reshape(data.labels.shape)
        preds = mode(preds, axis=-1).mode
        return np.squeeze(preds)
    
    def __str__(self) -> str:
        return "LDA"
    
class SVM(Model):
    def __init__(self):
        self.SVM = svm.SVC(gamma=0.001)
        
    def fit(self, data: Data):
        features = data.features.reshape((-1, data.features.shape[-1]))
        labels = data.labels.ravel()
        print(labels.shape)
        print(features.shape)
        self.SVM.fit(features, labels)

    def predict(self, data: Data):
        features = data.features.reshape((-1, data.features.shape[-1]))
        preds = self.SVM.predict(features)
        print(preds)
        preds = preds.reshape(data.labels.shape)
        preds = mode(preds, axis=-1).mode
        return np.squeeze(preds)
    
    def __str__(self) -> str:
        return "SVM"
    

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



@dataclass
class ModelConfig:
    state_size:          int
    hidden_layer_sizes:  list[int]
    num_residual_blocks: int
    residual_size:       int
    dropout:             float

class _BaseModel(abc.ABC, nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.build_model()

    @abc.abstractmethod
    def build_model(self):
        pass

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def activation_transform_layers(self, size: int) -> tuple[nn.Module, nn.Module, nn.Module]:
        return (
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(size),
            nn.Dropout(p=self.cfg.dropout, inplace=True),
        )

    def numel(self) -> int:
        """ Number of model parameters. Further docs here: https://pokemondb.net/pokedex/numel """
        return sum(p.numel() for p in self.parameters())

    def all_params(self) -> torch.Tensor:
        """ Returns an array of all model parameters """
        return torch.cat([x.detach().view(-1) for x in self.state_dict().values()])

class NNModel(_BaseModel):

    def build_model(self):

        # Build initial fully connected layers
        fully_connected = list()
        layer_sizes = [self.cfg.state_size] + self.cfg.hidden_layer_sizes + [self.cfg.residual_size]
        for in_size, out_size in zip(layer_sizes[:-1], layer_sizes[1:]):
            fully_connected.extend([
                nn.Linear(in_size, out_size),
                *self.activation_transform_layers(out_size),
            ])
        self.fully_connected = nn.Sequential(*fully_connected)

        # Build residual layers
        self.residual_blocks = nn.Sequential(*(
            _ResidualBlock(self.cfg) for _ in range(self.cfg.num_residual_blocks)
        ))

        # Final linear output layer
        self.output_layer = nn.Linear(self.cfg.residual_size, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fully_connected(x)
        x = self.residual_blocks(x)
        x = self.output_layer(x)
        return x

class _ResidualBlock(_BaseModel):

    num_layers = 2

    def build_model(self):
        fully_connected = list()
        for i in range(self.num_layers):
            fully_connected.append(
                nn.Linear(self.cfg.residual_size, self.cfg.residual_size)
            )
            if i < self.num_layers - 1:
                fully_connected.extend(
                    self.activation_transform_layers(self.cfg.residual_size)
                )

        self.fully_connected = nn.Sequential(*fully_connected)
        self.output_transform = nn.Sequential(*self.activation_transform_layers(self.cfg.residual_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fx = self.fully_connected(x)
        x = fx + x
        x = self.output_transform(x)
        return x



class DL(Model):

    def fit(self, data: Data):
        model_cfg = ModelConfig(
            data.num_features,
            [500, 200],
            0, 1000, 0.0)
        self.model = NNModel(model_cfg).to(device)
        self.model.train()

        epochs = 200
        batch_size = 300

        optim = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        crit = torch.nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim, T_0=epochs)

        features = torch.from_numpy(data.features)\
            .reshape(-1, data.num_features)\
            .float()\
            .to(device)
        labels = torch.from_numpy(data.labels)\
            .ravel()\
            .long()\
            .to(device) - 1

        for i in range(epochs):
            log.debug("Epoch %i / %i" % (i + 1, epochs))
            index = torch.randperm(len(features), device=device)
            batches = [index[j*batch_size:(j+1)*batch_size] for j in range(len(features)//batch_size)]
            losses = list()
            for j, batch_index in enumerate(batches):
                batch_data = features[batch_index]
                preds = self.model(batch_data)
                loss = crit(preds, labels[batch_index])
                losses.append(loss.item())
                loss.backward()
                optim.step()
                optim.zero_grad()
            log.debug("Mean loss: %.4f" % np.mean(losses))

            scheduler.step()

    @torch.no_grad()
    def predict(self, data: Data) -> np.ndarray:
        self.model.eval()
        features = torch.from_numpy(data.features)\
            .reshape(-1, data.num_features)\
            .float()\
            .to(device)

        preds = self.model(features).cpu().numpy()

        preds = np.argmax(preds, axis=-1) + 1
        preds = preds.reshape(data.labels.shape)
        preds = mode(preds, axis=2).mode
        return np.squeeze(preds)

    def __str__(self) -> str:
        return "NN"
