#Author: Gentry Atkinson
#Organization: Texas University
#Data: 21 July, 2021
#Train and test a CNN on the 6 datasets with their many label sets

import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Input
from tensorflow.keras.layers import Reshape, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

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

def build_cnn(X, num_classes, num_channels=1, opt='SGD', loss='mean_squared_error'):
    print("Input Shape: ", X.shape)
    model = Sequential([
        Input(shape=X[0].shape),
        Reshape((num_channels, X.shape[1])),
        BatchNormalization(scale=False),
        Conv1D(filters=128, kernel_size=16, activation='relu', padding='same'),
        MaxPooling1D(pool_size=(16), data_format='channels_first'),
        Conv1D(filters=128, kernel_size=16, activation='relu', padding='same'),
        MaxPooling1D(pool_size=(16), data_format='channels_first'),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=opt, loss=loss, metrics=['acc'])
    model.summary()
    return model

def train_cnn(model, X, y):
    es = EarlyStopping(monitor='val_acc', mode='max', verbose=1, patience=5)
    model.fit(X, y, epochs=100, verbose=1, callbacks=[es], validation_split=0.1, batch_size=10)

if __name__ == "__main__":
    print("Testing CNN")
    results_file = open('results/CNN_results', 'w+')
    X = np.genfromtxt('src/data/processed_datasets/ss1_attributes_train.csv', delimiter=',')
    y = np.genfromtxt('src/data/processed_datasets/ss1_labels_clean.csv', delimiter=',', dtype=int)
    y = to_categorical(y)
    model = build_cnn(X, 2)
    model = train_cnn(model, X, y)
