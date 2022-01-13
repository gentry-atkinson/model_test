#Author: Gentry Atkinson
#Organization: Texas State University
#Data: 13 January, 2022
#Get some accuracies to check sn data for suitability

import numpy as np
import tensorflow.keras.metrics as met
from tensorflow.keras import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Input
from tensorflow.keras.layers import Reshape, BatchNormalization, Dropout, ReLU, GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report
from sklearn.preprocessing import normalize
import gc
import os
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from utils.ts_feature_toolkit import calc_AER, calc_TER, calc_bias_metrics, calc_error_rates
from datetime import date

def build_cnn(X, num_classes, num_channels=1, opt='SGD', loss='mean_squared_error'):
    print("Input Shape: ", X.shape)
    model = Sequential([
        Input(shape=X[0].shape),
        Conv1D(filters=128, kernel_size=16, padding='same'),
        MaxPooling1D(pool_size=(2), data_format='channels_first'),
        Conv1D(filters=64, kernel_size=16, padding='same'),
        MaxPooling1D(pool_size=(2), data_format='channels_first'),
        Conv1D(filters=64, kernel_size=8, padding='same'),
        MaxPooling1D(pool_size=(2), data_format='channels_first'),
        Dropout(0.25),
        GlobalAveragePooling1D(),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=opt, loss=loss, metrics=[met.CategoricalAccuracy()])
    model.summary()
    return model

def train_cnn(model, X, y):
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)
    rlr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001)
    NUM_CORES = os.cpu_count()
    model.fit(X, y, epochs=500, verbose=1, callbacks=[es, rlr], validation_split=0.1, batch_size=32, workers=NUM_CORES)
    return model

def evaluate_cnn(model, X, y, mlr, base_fpr, base_fnr):
    y_pred = model.predict(X)
    y_pred = np.argmax(y_pred, axis=-1)
    y_true = np.argmax(y, axis=-1)
    print('Shape of y true: {}'.format(y_true.shape))
    print('Shape of y predicted: {}'.format(y_pred.shape))
    aer = calc_AER(y_true, y_pred)
    ter = calc_TER(aer, mlr)
    cev, sde = 0.0, 0.0
    print(base_fpr, base_fnr)
    if (base_fpr is None) or (base_fnr is None):
        pass
    else:
        fpr, fnr = calc_error_rates(y_true, y_pred)
        cev, sde = calc_bias_metrics(base_fpr, base_fnr, fpr, fnr)
    return classification_report(y_true, y_pred), confusion_matrix(y_true, y_pred), aer, ter, cev, sde

if __name__ == '__main__':
    print('Read Files')
    X_train = np.genfromtxt('src/data/processed_datasets/sn1_attributes_train.csv', delimiter=',')
    X_test = np.genfromtxt('src/data/processed_datasets/sn1_attributes_test.csv', delimiter=',')
    y_train = np.genfromtxt('src/data/processed_datasets/sn1_labels_clean.csv', delimiter=',', dtype=int)
    y_test = np.genfromtxt('src/data/processed_datasets/sn1_labels_test_clean.csv', delimiter=',', dtype=int)

    X_train = np.reshape(X_train, (len(X_train)//6, 6, 30))
    X_test = np.reshape(X_test, (len(X_test)//6, 6, 30))

    y_train = to_categorical(y_train)
    #y_test = to_categorical(y_test)

    print('Construct Model')
    mod = build_cnn(X_train, 2, num_channels=6, opt='ADAM', loss='binary_crossentropy')

    print('Train model')
    train_cnn(mod, X_train, y_train)

    print('Train labels: ', y_train)
    print('Test labels: ', y_test)

    print('Number of rainy test days: ', sum(y_test))
    print('Total days in test: ', len(y_test))

    y_pred = mod.predict(X_test)
    y_pred = np.argmax(y_pred, axis=-1)
    print('Predicted labels: ', y_pred)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
