#Author: Gentry Atkinson
#Organization: Texas University
#Data: 21 July, 2021
#Train and test a CNN on the 6 datasets with their many label sets

import numpy as np
import tensorflow.keras.metrics as met
from tensorflow.keras import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Input
from tensorflow.keras.layers import Reshape, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report

sets = [
    'bs1', 'bs2', 'har1', 'har2', 'ss1', 'ss2'
]

labels = [
    'labels_clean', 'ncar5', 'ncar10', 'nar5', 'nar10', 'nnar5', 'nnar10'
]

optimizers = [
    'SGD', 'RMSprop', 'adam'
]

losses = [
    'categorical_crossentropy', 'mean_squared_error', 'kullback_leibler_divergence'
]

chan_dic = {
    'bs1':1, 'bs2':2, 'har1':1, 'har2':3, 'ss1':1, 'ss2':1
}

class_dic = {
    'bs1':2, 'bs2':2, 'har1':7, 'har2':6, 'ss1':2, 'ss2':5
}

def build_cnn(X, num_classes, num_channels=1, opt='SGD', loss='mean_squared_error'):
    print("Input Shape: ", X.shape)
    model = Sequential([
        Input(shape=X[0].shape),
#        Reshape((num_channels, X.shape[2])),
        Conv1D(filters=128, kernel_size=32, activation='relu', padding='same'),
        MaxPooling1D(pool_size=(16), data_format='channels_first'),
        Conv1D(filters=128, kernel_size=16, activation='relu', padding='same'),
        MaxPooling1D(pool_size=(16), data_format='channels_first'),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=opt, loss=loss, metrics=[met.CategoricalAccuracy()])
    model.summary()
    return model

def train_cnn(model, X, y):
    es = EarlyStopping(monitor='val_categorical_accuracy', mode='max', verbose=1, patience=7)
    model.fit(X, y, epochs=100, verbose=1, callbacks=[es], validation_split=0.1, batch_size=10, workers=8)
    return model

def evaluate_cnn(model, X, y):
    y_pred = model.predict(X)
    y_pred = np.argmax(y_pred, axis=-1)
    y_true = np.argmax(y, axis=-1)
    return classification_report(y_true, y_pred)

if __name__ == "__main__":
    print("Testing CNN")
    f = 'bs2'
    results_file = open('results/CNN_results', 'w+')
    X = np.genfromtxt('src/data/processed_datasets/'+f+'_attributes_train.csv', delimiter=',')
    print("Shape of X: ", X.shape)
    NUM_INSTANCES = len(X)
    print("NUM_INSTANCES is ", NUM_INSTANCES)
    print("instances should be ", NUM_INSTANCES/chan_dic[f])
    SAMP_LEN = len(X[0])
    X = np.reshape(X, (int(NUM_INSTANCES/chan_dic[f]), chan_dic[f], SAMP_LEN))
    y = np.genfromtxt('src/data/processed_datasets/'+f+'_labels_clean.csv', delimiter=',', dtype=int)
    y = to_categorical(y)
    X_test = np.genfromtxt('src/data/processed_datasets/'+f+'_attributes_test.csv', delimiter=',')
    TEST_INSTANCES = len(X_test)
    X_test = np.reshape(X_test, (int(TEST_INSTANCES/chan_dic[f]), chan_dic[f], SAMP_LEN))
    y_test = np.genfromtxt('src/data/processed_datasets/'+f+'_labels_test.csv', delimiter=',', dtype=int)
    y_test = to_categorical(y_test)
    model = build_cnn(X, class_dic[f], opt='adam', loss='categorical_crossentropy')
    model = train_cnn(model, X, y)
    score = evaluate_cnn(model, X_test, y_test)
    print("Score for this model: \n", score)
