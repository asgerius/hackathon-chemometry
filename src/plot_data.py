import matplotlib.pyplot as plt
import numpy as np
import pelutils.ds.plots as plots

from data import load_dataframe, data_as_arrays, load_from_pickle


def plot_all_data(nm: np.ndarray, features: np.ndarray, labels: np.ndarray):

    labels = labels.ravel()
    features = features.reshape(-1, 700)
    with plots.Figure("data.png", figsize=(23, 10)):
        plt.subplot(131)
        plt.ylim([0, 1.8])
        plt.plot(nm, features[labels==1].T)
        plt.title("1 Treated")
        plt.grid()

        plt.subplot(132)
        plt.ylim([0, 1.8])
        plt.plot(nm, features[labels==2].T)
        plt.title("2 Virgin")
        plt.grid()

        plt.subplot(133)
        plt.ylim([0, 1.8])
        plt.plot(nm, features[labels==3].T)
        plt.title("3 Disposal")
        plt.grid()

if __name__ == "__main__":

    nm, features, labels = load_from_pickle()
    plot_all_data(nm, features, labels)
