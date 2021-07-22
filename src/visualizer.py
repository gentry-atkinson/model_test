#Author: Gentry Atkinson
#Organization: Texas University
#Data: 21 July, 2021
#Visualize 6 datasets with all their various label sets

import numpy as np
from sklearn.manifold import TSNE as tsne

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

    return X_avg
