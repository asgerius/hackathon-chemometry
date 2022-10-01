from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def load_dataframe() -> pd.DataFrame:
    return pd.read_csv("data.txt", delimiter="; ")

def data_as_arrays(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    num_scan = len(pd.unique(df.Scan))
    num_replicate = len(pd.unique(df.Replicate))
    num_sample = len(pd.unique(df.Sample))

    samples = pd.unique(df.Sample)
    sample_map = {int(v) - 1: i for i, v in enumerate(samples)}

    num_features = 700
    nm = np.array([int(x) for x in df.columns[4:]])
    features = np.empty((num_sample, num_replicate, num_scan, num_features))
    labels = np.empty((num_sample, num_replicate, num_scan), dtype=int)
    for i, row in df.iterrows():
        s = sample_map[int(row.Sample) - 1]
        r = int(row.Replicate) - 1
        sc = int(row.Scan) - 1
        features[s, r, sc] = row[df.columns[4:]].values
        labels[s, r, sc] = row.Type

    return nm, features, labels

@dataclass
class Data:
    nm: np.ndarray
    features: np.ndarray
    labels: np.ndarray

def save_to_pickle(data: Data):
    np.save("nm", data.nm)
    np.save("features", data.features)
    np.save("labels", data.labels)

def load_from_pickle() -> Data:
    try:
        return Data(
            nm = np.load("nm.npy"),
            features = np.load("features.npy"),
            labels = np.load("labels.npy"),
        )
    except FileNotFoundError:
        df = load_dataframe()
        nm, features, labels = data_as_arrays(df)
        save_to_pickle(Data(nm, features, labels))
        return load_from_pickle()
