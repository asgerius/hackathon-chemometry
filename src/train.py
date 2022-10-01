from __future__ import annotations

import os
import pickle
from argparse import ArgumentParser

import numpy as np
from pelutils import log, LogLevels
from sklearn.model_selection import KFold

import model as model_module
from data import load_from_pickle
from preprocess import apply_combined_linear, apply_derivatives, apply_logs, apply_standardize, combined_linear, get_derivatives, get_logarithm, standardize


def cv(path: str, model_name: str, num_splits: int, preprocessing_steps: list[str]):

    log.section("Loading data")
    data = load_from_pickle()

    log.section("Preprocessing")
    log("Preprocessing steps:", *preprocessing_steps)

    # log_data = get_logarithm(data)
    # data.features = apply_logs(data.features, log_data)
    # deriv_data = get_derivatives(data)
    # data.features = apply_derivatives(data.features, deriv_data)

    if "standardize" in preprocessing_steps:
        preprocessing_steps.remove("standardize")
        mu, std = standardize(data)
        apply_standardize(data, mu, std)
        with open(f"{path}/standardize.pkl", "wb") as f:
            pickle.dump((mu, std), f)
    if "linreg" in preprocessing_steps:
        preprocessing_steps.remove("linreg")
        lr = combined_linear(data)
        apply_combined_linear(data, lr)
        with open(f"{path}/combined_linear.pkl", "wb") as f:
            pickle.dump(lr, f)

    if preprocessing_steps:
        raise ValueError(f"The following preprocessing steps were not used: {preprocessing_steps}")

    kfold = KFold(num_splits, shuffle=True)
    accs = list()

    for i, (train_index, test_index) in enumerate(kfold.split(data.features)):
        log.section("Split %i / %i" % (i + 1, num_splits))
        train_data = data.split_by_index(train_index)
        test_data = data.split_by_index(test_index)

        model: model_module.Model = getattr(model_module, model_name)()
        log("Got model %s" % model)

        log("Fitting")
        model.fit(train_data)

        log("Predicting")
        preds = model.predict(test_data)
        labels = test_data.labels[:, :, 0]

        correctly_classified = (preds == labels).sum()
        total = preds.size
        log(
            "Correctly classified: %i / %i" % (correctly_classified, total),
            "%.2f %%" % (100 * correctly_classified / total),
        )
        accs.append(correctly_classified / total)

    log.section("Done classifying")
    log("Mean accuracy: %.2f %%" % (100 * np.mean(accs)))

    log.section("Retraining on entire dataset")

    model: model_module.Model = getattr(model_module, model_name)()
    log("Got model %s" % model)

    log("Fitting")
    with log.no_log:
        model.fit(data)

    log("Saving model")
    model.save(path)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("model_name")
    parser.add_argument("-n", "--num_splits", type=int, default=2)
    parser.add_argument("-p", "--preprocessing", nargs="*", default=list())
    args = parser.parse_args()

    dir = os.path.join("out", args.name)
    os.makedirs(dir, exist_ok=True)

    log.configure(f"{dir}/train.log", print_level=LogLevels.DEBUG)

    cv(dir, args.model_name, args.num_splits, args.preprocessing)
