from __future__ import annotations

import os
import pickle
from argparse import ArgumentParser

import numpy as np
from pelutils import log, LogLevels
from sklearn.model_selection import KFold

import model as model_module
from data import Data, load_from_pickle
from preprocess import apply_combined_linear, apply_standardize, apply_within, combined_linear, get_within, standardize


def cv(path: str):

    log("Load")

    with open(f"{path}/model.pkl", "rb") as f:
        model: model_module.Model = pickle.load(f)

    with open(f"{path}/preprocess.pkl", "rb") as f:
        preprocessing_steps: list[str] = pickle.load(f)

    log("Data")

    data = load_from_pickle()

    log("Preprocessing", preprocessing_steps)

    if "within" in preprocessing_steps:
        preprocessing_steps.remove("within")
        m, s = get_within(data)
        apply_within(data, m, s)
    if "standardize" in preprocessing_steps:
        preprocessing_steps.remove("standardize")
        with open(f"{path}/standardize.pkl", "rb") as f:
            mu, std = pickle.load(f)
        apply_standardize(data, mu, std)

    if preprocessing_steps:
        raise ValueError(f"The following preprocessing steps were not used: {preprocessing_steps}")

    preds = model.predict(data).ravel()
    log(np.unique(preds, return_counts=True)[1])
    with open(f"{path}/preds.csv", "w", encoding="ascii") as f:
        preds_iter = iter(preds)
        f.write("Sample,Replicate,Type\n")
        for i in range(201, 201 + len(preds) // 3):
            for j in range(1, 4):
                f.write("%i,%i,%i\n" % (i, j, next(preds_iter)))

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("name")
    args = parser.parse_args()

    dir = os.path.join("out", args.name)

    log.configure(f"{dir}/eval.log", print_level=LogLevels.DEBUG)

    with log.log_errors:
        cv(dir)
