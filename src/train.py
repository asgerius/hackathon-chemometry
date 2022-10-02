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


def cv(path: str, model_name: str, num_splits: int, preprocessing_steps: list[str]):

    log.section("Loading data")
    data = load_from_pickle()
    log(data.features.shape)

    log.section("Preprocessing")
    log("Preprocessing steps:", *preprocessing_steps)

    with open(f"{path}/preprocess.pkl", "wb") as f:
        pickle.dump(preprocessing_steps, f)

    mis_assignments = np.zeros(3, dtype=int)
    assignments = np.zeros((3, 3), dtype=int)

    if "within" in preprocessing_steps:
        preprocessing_steps.remove("within")
        m, s = get_within(data)
        apply_within(data, m, s)
        with open(f"{path}/apply-within.pkl", "wb") as f:
            pickle.dump((m, s), f)
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
        log.debug(train_index.tolist(), test_index.tolist())
        log.section("Split %i / %i" % (i + 1, num_splits))
        train_data = data.split_by_index(train_index)
        test_data = data.split_by_index(test_index)
        log(train_data.features.shape)
        log(test_data.features.shape)

        model: model_module.Model = getattr(model_module, model_name)()
        log("Got model %s" % model)

        log("Fitting")
        model.fit(train_data)

        log("Predicting")
        preds = model.predict(test_data).ravel()
        labels = test_data.labels[:, :, 0].ravel()

        for i in range(data.num_labels):
            p = preds[labels==i+1]
            log.debug(
                "Label %i" % (i + 1),
                "%i / %i, %.2f %%" % ((p == i + 1).sum(), p.size, 100 * (p==i+1).sum() / p.size),
            )
            mis_assignments[i] += (p != i + 1).sum()
            p = np.append(p, 1)
            p = np.append(p, 2)
            p = np.append(p, 3)
            assignments[i] += np.unique(p, return_counts=True)[1]
            assignments[i] -= 1

        correctly_classified = (preds == labels).sum()
        total = preds.size
        log(
            "Correctly classified: %i / %i" % (correctly_classified, total),
            "%.2f %%" % (100 * correctly_classified / total),
        )
        accs.append(correctly_classified / total)

    log("Misclassifications", mis_assignments.tolist(), mis_assignments / mis_assignments.sum())
    log("Assignments", assignments)

    log.section("Done classifying")
    log("Mean accuracy: %.2f %%" % (100 * np.mean(accs)))

    log.section("Retraining on entire dataset")

    model: model_module.Model = getattr(model_module, model_name)()
    log("Got model %s" % model)

    log("Fitting")
    with log.no_log:
        pass
        # model.fit(data)

    log("Saving model")
    model.save(path)

def cv_twostage(path: str, model_name: str, num_splits: int, preprocessing_steps: list[str]):

    log.section("Loading data")
    data = load_from_pickle()
    log(data.features.shape)

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

    m, s = get_within(data)
    apply_within(data, m, s)

    if preprocessing_steps:
        raise ValueError(f"The following preprocessing steps were not used: {preprocessing_steps}")

    kfold = KFold(num_splits, shuffle=True)
    accs = list()

    for i, (train_index, test_index) in enumerate(kfold.split(data.features)):
        log.section("Split %i / %i" % (i + 1, num_splits))

        log("Train treated / not treated")
        train_data = data.split_by_index(train_index).treated_labels()
        test_data = data.split_by_index(test_index).treated_labels()
        log(train_data.features.shape)
        log(test_data.features.shape)

        model1: model_module.Model = getattr(model_module, model_name)()
        log("Got model %s" % model1)

        log("Fitting")
        model1.fit(train_data)

        log("Predicting")
        preds = model1.predict(test_data)
        labels = test_data.labels[:, :, 0]

        correctly_classified = (preds == labels).sum()
        total = preds.size
        log(
            "Correctly classified: %i / %i" % (correctly_classified, total),
            "%.2f %%" % (100 * correctly_classified / total),
        )

        log("Train virgin / disposal")
        train_data = data.split_by_index(train_index).no_treated()
        preds = preds.ravel()
        log(train_data.features.shape)
        test_data = Data(
            data.nm,
            test_data.features.reshape(-1, *test_data.features.shape[2:])[preds==2],
            test_data.labels.reshape(-1, *test_data.labels.shape[2:])[preds==2],
        )
        log(test_data.features.shape)

        model2: model_module.Model = getattr(model_module, model_name)()
        model2.fit(train_data)
        preds[preds==2] = model2.predict(test_data)
        preds = preds.reshape(-1, 3)
        labels = data.split_by_index(test_index).labels[:, :, 0]

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

    model1: model_module.Model = getattr(model_module, model_name)()
    log("Got model %s" % model1)

    log("Fitting")
    with log.no_log:
        pass
        # model.fit(data)

    log("Saving model")
    model1.save(path)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("model_name")
    parser.add_argument("-n", "--num_splits", type=int, default=2)
    parser.add_argument("-p", "--preprocessing", nargs="*", default=list())
    parser.add_argument("-t", "--two_stage", action="store_true")
    args = parser.parse_args()

    dir = os.path.join("out", args.name)
    os.makedirs(dir, exist_ok=True)

    log.configure(f"{dir}/train.log", print_level=LogLevels.DEBUG)

    with log.log_errors:
        (cv_twostage if args.two_stage else cv)(dir, args.model_name, args.num_splits, args.preprocessing)
