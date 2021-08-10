#Author: Gentry Atkinson
#Organization: Texas University
#Data: 21 July, 2021
#Visualize 6 datasets with all their various label sets

import numpy as np
from sklearn.manifold import TSNE as tsne
from sklearn.preprocessing import normalize
from matplotlib import pyplot as plt
from utils.color_pal import color_pallette_small
from utils.ts_feature_toolkit import get_features_for_set

sets = [
    'bs1', 'bs2', 'har1', 'har2', 'ss1', 'ss2'
]

labels = [
    'labels_clean', 'ncar5', 'ncar10', 'nar5', 'nar10', 'nnar5', 'nnar10'
]

chan_dic = {
    'bs1':1, 'bs2':2, 'har1':1, 'har2':3, 'ss1':1, 'ss2':1
}

class_dic = {
    'bs1':2, 'bs2':2, 'har1':7, 'har2':6, 'ss1':2, 'ss2':5
}

def avgChannels(X, num_channels):
    X_avg = np.zeros((len(X)//num_channels, len(X[0])))
    for i in range(0, len(X_avg)):
        X_avg[i, :] = np.sum(X[num_channels*i:num_channels*i+num_channels, :], axis=0)
        X_avg[i, :] /= num_channels
    return X_avg

if __name__ == "__main__":
    print("Making Pictures")

    f = 'ss1'

    X = np.genfromtxt('src/data/processed_datasets/'+f+'_attributes_train.csv', delimiter=',')
    print("Shape of X: ", X.shape)
    NUM_INSTANCES = len(X)
    print("NUM_INSTANCES is ", NUM_INSTANCES)
    print("instances should be ", NUM_INSTANCES/chan_dic[f])
    SAMP_LEN = len(X[0])

    X = normalize(X, norm='max')
    X = avgChannels(X, chan_dic[f])
    #transform = np.real(np.fft.rfft2(X))
    transform = get_features_for_set(X, num_samples=len(X))
    print("Size of FFT transform: ", transform.shape)
    print(transform)
    y = np.genfromtxt('src/data/processed_datasets/'+f+'_labels_clean.csv', delimiter=',', dtype=int)

    vis = tsne(n_components=2).fit_transform(transform)

    for i in range(class_dic[f]):
        plt.scatter(vis[np.where(y==i), 0], vis[np.where(y==i), 1], s=2, c=color_pallette_small[i])
    plt.show()
