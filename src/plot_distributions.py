

import os

import matplotlib.pyplot as plt
import numpy as np
import pelutils.ds.plots as plots

from data import Data, load_from_pickle


_labels = "1 Treated", "2 Virgin", "3 Disposal"

def t(arr: np.ndarray) -> np.ndarray:
    return arr
    # return np.log(arr)
    # return np.log(np.sqrt(arr))

def plot_at_wavelength(data: Data):
    wl = 2150
    wl_index = np.where((data.nm==wl))[0][0]

    features = data.features[..., wl_index].ravel()
    labels = data.labels.ravel()

    fs = [features[labels==x] for x in (1, 2, 3)]

    with plots.Figure("wl-plots/wl-%i" % wl):
        for i in range(3):
            plt.plot(*plots.get_bins(t(fs[i]), bins=40), color=plots.tab_colours[i], label=_labels[i])

        plt.title("$\lambda=%i$ nm" % wl)
        plt.grid()
        plt.legend()

    features = data.features[..., wl_index].reshape(len(data), -1)
    labels = data.labels[:, 0, 0]

    with plots.Figure("wl-plots/samples-wl-%i" % wl):
        for i in range(len(features)):
            plt.plot(*plots.get_bins(t(features[i]), bins=7), color=plots.tab_colours[labels[i]-1], alpha=0.5)

        plt.title("$\lambda=%i$ nm" % wl)
        plt.grid()

if __name__ == "__main__":
    os.makedirs("wl-plots", exist_ok=True)
    data = load_from_pickle()
    plot_at_wavelength(data)
