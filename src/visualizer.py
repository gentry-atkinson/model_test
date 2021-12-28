#Author: Gentry Atkinson
#Organization: Texas State University
#Data: 21 July, 2021
#Visualize 6 datasets with all their various label sets

import numpy as np
from sklearn.manifold import TSNE as tsne
from sklearn.preprocessing import normalize
from matplotlib import pyplot as plt
from utils.color_pal import color_pallette_small, color_pallette_big
from utils.ts_feature_toolkit import get_features_for_set
#from fastdist import fastdist
import gc

# sets = [
#     'bs1', 'bs2', 'har1', 'har2', 'ss1', 'ss2'
# ]

sets = [
    'har2'
]

labels = [
    'clean', 'ncar5', 'ncar10', 'nar5', 'nar10', 'nnar5', 'nnar10'
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

def absChannels(X, num_channels):
    X_avg = np.zeros((len(X)//num_channels, len(X[0])))
    for i in range(0, len(X_avg)):
        X_avg[i, :] = np.linalg.norm(X[num_channels*i:num_channels*i+num_channels, :], axis=0)
    return X_avg

RUN_TSNE = True
RUN_WF = False

if __name__ == "__main__":
    print("Making Pictures")
    if RUN_TSNE:
        for f in sets:
            print("Set: ", f)
            X = np.genfromtxt('src/data/processed_datasets/'+f+'_attributes_train.csv', delimiter=',')
            print("Shape of X: ", X.shape)
            NUM_INSTANCES = len(X)
            print("NUM_INSTANCES is ", NUM_INSTANCES)
            print("instances should be ", NUM_INSTANCES/chan_dic[f])
            SAMP_LEN = len(X[0])
            X = normalize(X, norm='max')
            X = absChannels(X, chan_dic[f])
            transform = get_features_for_set(X)
            print("Size of feature set: ", transform.shape)
            vis = tsne(n_components=2, metric='euclidean', n_iter_without_progress=100).fit_transform(transform)

            if class_dic[f] > 5:
                pal = color_pallette_big
            else:
                pal = color_pallette_small

            for l in labels:
                y = np.genfromtxt('src/data/processed_datasets/'+f+'_labels_'+l+'.csv', delimiter=',', dtype=int)

                plt.figure()
                plt.axis('off')
                for i in range(class_dic[f]):
                    plt.scatter(vis[np.where(y==i), 0], vis[np.where(y==i), 1], s=2, c=pal[i], label=str(i))
                #plt.legend()
                plt.savefig("imgs/"+f+"_"+l+"_tsne.pdf")
                gc.collect()
                plt.close()

    if RUN_WF:
        for f in sets:
            print("Set: ", f)
            X = np.genfromtxt('src/data/processed_datasets/'+f+'_attributes_train.csv', delimiter=',')
            y = np.genfromtxt('src/data/processed_datasets/'+f+'_labels_clean.csv', delimiter=',', dtype=int)
            X = normalize(X, norm='max')
            X = absChannels(X, chan_dic[f])
            NUM_INSTANCES = len(X)
            print("NUM_INSTANCES is ", NUM_INSTANCES)
            SAMP_LEN = len(X[0])
            if class_dic[f] > 5:
                pal = color_pallette_big
            else:
                pal = color_pallette_small

            plt.figure()
            for l in range(class_dic[f]):
                if np.sum(y==l) != 0:
                    i = X[np.where(y==l)][0]
                    plt.plot(range(SAMP_LEN), i, c=pal[l])
            plt.savefig("imgs/"+f+"_waveforms.pdf")
            gc.collect()
            plt.close()
