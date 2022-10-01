import matplotlib.pyplot as plt
import numpy as np
import pelutils.ds.plots as plots

from data import Data, load_from_pickle
from preprocess import combined_linear, apply_combined_linear


def plot_all_data(data: Data):

    labels = data.labels.ravel()
    features = data.features.reshape(-1, 700)
    with plots.Figure("data.png", figsize=(23, 10)):
        plt.subplot(131)
        plt.ylim([-0.5, 1])
        plt.plot(data.nm, features[labels==1].T)
        plt.title("1 Treated")
        plt.grid()

        plt.subplot(132)
        plt.ylim([-0.5, 1])
        plt.plot(data.nm, features[labels==2].T)
        plt.title("2 Virgin")
        plt.grid()

        plt.subplot(133)
        plt.ylim([-0.5, 1])
        plt.plot(data.nm, features[labels==3].T)
        plt.title("3 Disposal")
        plt.grid()

    with plots.Figure("data-means.png", figsize=(20, 10)):
        m1 = features[labels==1].mean(axis=0)
        m2 = features[labels==2].mean(axis=0)
        m3 = features[labels==3].mean(axis=0)

        std1 = features[labels==1].std(axis=0)
        std2 = features[labels==2].std(axis=0)
        std3 = features[labels==3].std(axis=0)

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

if __name__ == "__main__":

    data = load_from_pickle()
    lr = combined_linear(data)
    apply_combined_linear(data, lr)
    plot_all_data(data)
