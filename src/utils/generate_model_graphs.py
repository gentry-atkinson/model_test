#Author: Gentry Atkinson
#Organization: Texas University
#Data: 07 January, 2021
#Print model graphs from keras

import tensorflow as tf
from tensorflow import keras
import tensorboard
from build_AE import build_AE
from scipy.io import loadmat
import numpy as np
from datetime import datetime

def get_unimib_data(s="acc"):
    print("Loading UniMiB set ", s)
    X_flat = loadmat("data/UniMiB-SHAR/data/" + s + "_data.mat")[s + "_data"]
    y = loadmat("data/UniMiB-SHAR/data/" + s + "_labels.mat")[s + "_labels"][:,0]
    if(s=="acc"):
        labels = loadmat("data/UniMiB-SHAR/data/" + s + "_names.mat")[s + "_names"][0,:]
    else:
        labels = loadmat("data/UniMiB-SHAR/data/" + s + "_names.mat")[s + "_names"][:,0]
    print("Num instances: ", len(X_flat))
    print("Instance length: ", len(X_flat[0]))

    y = np.array(y - 1)
    X = np.zeros((len(X_flat), 3, 151), dtype='float')
    X[:,0,0:151]=X_flat[:,0:151]
    X[:,1,0:151]=X_flat[:,151:302]
    X[:,2,0:151]=X_flat[:,302:453]
    print(labels)
    return X, y, labels

if __name__ == "__main__":
    print("Trained Autoencoder")
    X, y, labels = get_unimib_data("adl")
    model = build_AE(X[0])

    logdir="logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

    model.fit(X, X, callbacks=[tensorboard_callback], epochs=5)
