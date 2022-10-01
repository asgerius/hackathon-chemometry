from __future__ import annotations

import os
from argparse import ArgumentParser

import numpy as np
from pelutils import log, LogLevels
from sklearn.model_selection import KFold

import model as model_module
from data import load_from_pickle


def cv(path: str, model_name: str, num_splits: int):

    log("Loading data")
    data = load_from_pickle()

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

        corretly_classified = (preds == labels).sum()
        total = preds.size
        log(
            "Correctly classified: %i / %i" % (corretly_classified, total),
            "%.2f %%" % (100 * corretly_classified / total),
        )
        accs.append(corretly_classified / total)

    log.section("Done classifying")
    log("Mean accuracy: %.2f %%" % (100 * np.mean(accs)))

    log.section("Retraining on entire dataset")

    model: model_module.Model = getattr(model_module, model_name)()
    log("Got model %s" % model)

    log("Fitting")
    model.fit(data)

    log("Saving model")
    model.save(path)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("model_name")
    parser.add_argument("-n", "--num_splits", type=int, default=2)
    args = parser.parse_args()

    dir = os.path.join("out", args.name)
    os.makedirs(dir, exist_ok=True)

    log.configure(f"{dir}/train.log", print_level=LogLevels.DEBUG)

    cv(dir, args.model_name, args.num_splits)
