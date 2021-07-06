#Author: Gentry Atkinson
#Organization: Texas University
#Data: 29 October, 2020
#Build, compile and return a convolutional autoencoder

#from tensorflow import keras
import numpy as np
import random as rand
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dropout, Dense
from tensorflow.keras.layers import Input, Conv1DTranspose, Lambda, Reshape, BatchNormalization
from tensorflow.keras.layers import UpSampling1D
from tensorflow.keras.callbacks import EarlyStopping
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

#tf.config.threading.set_inter_op_parallelism_threads(8)

def build_sfe(train_example, features_per_channel=30, num_labels=2):
    print("Input shape: ", train_example.shape)
    s=[1,1]
    if train_example.ndim==1:
        s[1] = len(train_example)
    else:
        s[0] = train_example.shape[0]
        s[1] = train_example.shape[1]
    model = Sequential([
        Input(shape=train_example.shape),
        BatchNormalization(scale=False),
        Reshape((s[0], s[1])),
        Conv1D(filters=128, kernel_size=16, activation='relu', padding='same'),
        Conv1D(filters=128, kernel_size=16, activation='relu', padding='same'),
        MaxPooling1D(pool_size=(16), data_format='channels_first'),
        Flatten(),
        Dense(features_per_channel*s[0], activation='sigmoid', name='Embedding'),
        Dense(features_per_channel*s[0], activation='sigmoid'),
        Dense(features_per_channel*s[0]/2, activation='sigmoid'),
        Dense(num_labels, activation='softmax')
    ])
    model.compile(optimizer='RMSprop', loss='mse', metrics=['acc'])
    model.summary()
    print("Output shape: ", model.output.shape)
    return model

def trim_classifier(sfe):
    print("Removing classification layers")
    o = sfe.layers[-4].output
    extractor = Model(sfe.input, [o])
    #extractor.summary()
    return extractor

def train_sfe(sfe, X, y, withEvaluation=False):
    if withEvaluation:
        X, X_test, y, y_test = train_test_split(X, y, test_size=0.1, random_state=23)

    es = EarlyStopping(monitor='val_acc', mode='max', verbose=1, patience=5)
    sfe.fit(X, y, epochs=100, verbose=1, callbacks=[es], validation_split=0.1, batch_size=10)

    if withEvaluation:
        #y_pred = sfe.predict_classes(X_test)
        y_pred = np.argmax(sfe.predict(X_test), axis=-1)
        y_test = np.argmax(y_test, axis=-1)
        mat = confusion_matrix(y_test, y_pred)
        print("predicted labels ", y_pred)
        print("true labels ", y_test)
        print("Confuxion matrix: ")
        print(mat)

    return sfe

def get_trained_sfe(X, y):
    if y.ndim == 1:
        y = to_categorical(y)
    numLabels = y.shape[1]
    sfe = build_sfe(X[0], num_labels=numLabels)
    sfe = train_sfe(sfe, X, y)
    sfe = trim_classifier(sfe)
    sfe.summary()
    return sfe

if __name__ == "__main__":
    print("Test model building")
    X = np.genfromtxt('data/synthetic_test_data.csv', delimiter=',')
    y = np.genfromtxt('data/synthetic_test_labels.csv', delimiter=',')
    y = to_categorical(y)
    print("data shape: ", X.shape)
    print("label shape: ", y.shape)
    norm = np.max(X)
    X = X/norm
    X = np.resize(X, (len(X), 1, len(X[0])))
    print("labels", y)

    sfe = build_sfe(X[0], num_labels=3)
    sfe = train_sfe(sfe, X, y, withEvaluation=True)
    sfe = trim_classifier(sfe)
    predict = sfe.predict(X)
    print("Output feature set: ", predict.shape)
