import matplotlib.pyplot as plt
import numpy as np
import pelutils.ds.plots as plots

from data import Data, load_from_pickle
from preprocess import apply_standardize, combined_linear, apply_combined_linear, standardize


def plot_all_data(data: Data):

    labels = data.labels.ravel()
    features = data.features.reshape(-1, 700)
    f1 = features[labels==1]
    f2 = features[labels==2]
    f3 = features[labels==3]

    with plots.Figure("data.png", figsize=(23, 10)):
        plt.subplot(131)
        plt.ylim([-3, 5])
        plt.plot(data.nm, f1.T)
        plt.title("1 Treated")
        plt.grid()

        plt.subplot(132)
        plt.ylim([-3, 5])
        plt.plot(data.nm, f2.T)
        plt.title("2 Virgin")
        plt.grid()

        plt.subplot(133)
        plt.ylim([-3, 5])
        plt.plot(data.nm, f3.T)
        plt.title("3 Disposal")
        plt.grid()

    with plots.Figure("data-means.png", figsize=(20, 10)):
        m1 = f1.mean(axis=0)
        m2 = f2.mean(axis=0)
        m3 = f3.mean(axis=0)

        std1 = f1.std(axis=0)
        std2 = f2.std(axis=0)
        std3 = f3.std(axis=0)

        plt.subplot(121)
        plt.plot(data.nm, m1, label="1 Treated")
        plt.plot(data.nm, m2, label="2 Virgin")
        plt.plot(data.nm, m3, label="3 Disposal")

        plt.title("Mean value by class")
        plt.grid()
        plt.legend()

        plt.subplot(122)
        plt.plot(data.nm, std1, label="1 Treated")
        plt.plot(data.nm, std2, label="2 Virgin")
        plt.plot(data.nm, std3, label="3 Disposal")

        plt.title("Std. by class")
        plt.grid()
        plt.legend()

    num_samples = 4
    f1_sub = f1.reshape(-1, *data.features.shape[1:])[5:5+num_samples]
    f2_sub = f2.reshape(-1, *data.features.shape[1:])[5:5+num_samples]
    f3_sub = f3.reshape(-1, *data.features.shape[1:])[5:5+num_samples]

    with plots.Figure("data-samples.png", figsize=(23, 10)):
        plt.subplot(131)
        plt.ylim([-3, 5])
        for i in range(num_samples):
            plt.plot(data.nm, f1_sub[i].reshape((-1, 700)).T, color=plots.tab_colours[i], lw=0.5)
        plt.title("1 Treated")
        plt.grid()

        plt.subplot(132)
        plt.ylim([-3, 5])
        for i in range(num_samples):
            plt.plot(data.nm, f2_sub[i].reshape((-1, 700)).T, color=plots.tab_colours[i], lw=0.5)
        plt.title("2 Virgin")
        plt.grid()

        plt.subplot(133)
        plt.ylim([-3, 5])
        for i in range(num_samples):
            plt.plot(data.nm, f3_sub[i].reshape((-1, 700)).T, color=plots.tab_colours[i], lw=0.5)
        plt.title("3 Disposal")
        plt.grid()

    f1_within_sample_std = f1.reshape(-1, *data.features.shape[1:]).std(axis=(1, 2))
    f2_within_sample_std = f2.reshape(-1, *data.features.shape[1:]).std(axis=(1, 2))
    f3_within_sample_std = f3.reshape(-1, *data.features.shape[1:]).std(axis=(1, 2))
    lower, upper = -2, 2
    with plots.Figure("data-within-sample-variance.png", figsize=(23, 10)):
        plt.subplot(131)
        plt.ylim([lower, upper])
        plt.plot(data.nm, f1_within_sample_std.T, color="grey", alpha=0.5)
        plt.title("1 Treated")
        plt.grid()

        plt.subplot(132)
        plt.ylim([lower, upper])
        plt.plot(data.nm, f2_within_sample_std.T, color="grey", alpha=0.5)
        plt.title("2 Virgin")
        plt.grid()

        plt.subplot(133)
        plt.ylim([lower, upper])
        plt.plot(data.nm, f3_within_sample_std.T, color="grey", alpha=0.5)
        plt.title("3 Disposal")
        plt.grid()

if __name__ == "__main__":

    data = load_from_pickle()
    # mu, std = standardize(data)
    # apply_standardize(data, mu, std)
    # lr = combined_linear(data)
    # apply_combined_linear(data, lr)
    plot_all_data(data)
