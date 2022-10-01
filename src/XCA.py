import matplotlib.pyplot as plt
import numpy as np
import pelutils.ds.plots as plots
from sklearn.decomposition import FastICA, PCA
import pickle

from data import Data, load_from_pickle
from preprocess import combined_linear, apply_combined_linear
from src.preprocess import apply_standardize, standardize

def PCAmaker(data: Data):
    mu, std = standardize(data)
    apply_standardize(data, mu, std)
    features = data.features.reshape(-1, 700)
    #print(np.shape(features))
    pcatransformer = PCA(whiten=True)
    pcatransformer.fit(features)
    pcaout = pcatransformer.fit_transform(features)
    print(pcatransformer.explained_variance_ratio_)
    plt.scatter(pcaout[:,0],pcaout[:,1])
    plt.show()


def ICA(data: Data):
    features = data.features.reshape(-1, 700)
    #print(np.shape(features))
    icatransformer = FastICA()
    icatransformer.fit(features)
    path = r"C:\Users\Tjalf\OneDrive - Danmarks Tekniske Universitet\DSK Hack\hackathon-chemometry\XCA"
    path = f"{path}/ICA.pkl"
    with open(path, "wb") as f:
        pickle.dump(icatransformer,f)

    
    

    plt.show()



if __name__ == "__main__":

    data = load_from_pickle()
    lr = combined_linear(data)
    apply_combined_linear(data, lr)
    ICA(data)
  