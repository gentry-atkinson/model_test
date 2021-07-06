#Author: Gentry Atkinson
#Organization: Texas University
#Data: 31 October, 2020
#Train and return a deep neural network

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

def build_dnn(train_example, num_labels=2):
    print("Input shape: ", train_example.shape)
    model = Sequential([
        Input(shape=train_example.shape),
        BatchNormalization(scale=False),
        Dense(128, activation='relu'),
        Dense(128, activation='relu'),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(num_labels, activation='softmax')
    ])
    model.compile(optimizer='RMSprop', loss='categorical_crossentropy', metrics=['acc'])
    model.summary()
    print("Output shape: ", model.output.shape)
    return model

def train_dnn(dnn, X, y, withEvaluation=False):
    if withEvaluation:
        X, X_test, y, y_test = train_test_split(X, y, test_size=0.1, random_state=23)

    es = EarlyStopping(monitor='val_acc', mode='max', verbose=1, patience=4)
    dnn.fit(X, y, epochs=100, verbose=1, callbacks=[es], validation_split=0.1, batch_size=10)

    if withEvaluation:
        y_pred = np.argmax(dnn.predict(X_test), axis=-1)
        y_test = np.argmax(y_test, axis=-1)
        mat = confusion_matrix(y_test, y_pred)
        print("predicted labels ", y_pred)
        print("true labels ", y_test)
        print("Confuxion matrix: ")
        print(mat)

    return dnn

def get_trained_dnn(X, y):
    numLabels = y.shape[1]
    dnn = build_dnn(X[0], num_labels=numLabels)
    dnn = train_dnn(dnn, X, y, withEvaluation=True)
    return dnn

if __name__ == "__main__":
    print("Test model building")
    X = np.genfromtxt('data/synthetic_test_data.csv', delimiter=',')
    y = np.genfromtxt('data/synthetic_test_labels.csv', delimiter=',')
    y = to_categorical(y)
    print("data shape: ", X.shape)
    print("label shape: ", y.shape)
    norm = np.max(X)
    X = X/norm
    #X = np.resize(X, (len(X), 1, len(X[0])))
    print("labels", y)

    dnn = build_dnn(X[0], num_labels=3)
    dnn = train_dnn(dnn, X, y, withEvaluation=True)
    predict = dnn.predict(X)
    print("Output feature set: ", predict.shape)
