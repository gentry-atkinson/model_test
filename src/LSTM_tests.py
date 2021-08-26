#Author: Gentry Atkinson
#Organization: Texas University
#Data: 23 August, 2021
#Train and test a CNN on the 6 datasets with their many label sets

import numpy as np
import tensorflow.keras.metrics as met
from tensorflow.keras import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import LSTM, Input, Dense
from tensorflow.keras.layers import Reshape, BatchNormalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report
from sklearn.preprocessing import normalize
import gc
from sklearn.utils import shuffle

sets = [
    'bs1', 'bs2', 'har1', 'har2', 'ss1', 'ss2'
]

labels = [
    'labels_clean', 'labels_ncar5', 'labels_ncar10', 'labels_nar5',
    'labels_nar10', 'labels_nnar5', 'labels_nnar10'
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
        LSTM(32),
        Dropout(0.25),
        Dense(128, activation='relu'),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=opt, loss=loss, metrics=[met.CategoricalAccuracy()])
    model.summary()
    return model

def train_cnn(model, X, y):
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=7)
    model.fit(X, y, epochs=100, verbose=1, callbacks=[es], validation_split=0.1, batch_size=100, workers=8)
    return model

def evaluate_cnn(model, X, y):
    y_pred = model.predict(X)
    y_pred = np.argmax(y_pred, axis=-1)
    y_true = np.argmax(y, axis=-1)
    return classification_report(y_true, y_pred)

if __name__ == "__main__":
    print("Testing LSTM")
    results_file = open('results/LSTM_results.txt', 'w+')
    counter = 1
    for f in sets:
        for l in labels:
            print ('Experiment: ', counter, " Set: ", f, "Labels: ", l)
            results_file.write('############Experiment {}############\n'.format(counter))
            results_file.write('Set: {}\n'.format(f))
            results_file.write('Labels: {}\n'.format(l))
            X = np.genfromtxt('src/data/processed_datasets/'+f+'_attributes_train.csv', delimiter=',')
            print("Shape of X: ", X.shape)
            NUM_INSTANCES = len(X)
            print("NUM_INSTANCES is ", NUM_INSTANCES)
            print("instances should be ", NUM_INSTANCES/chan_dic[f])
            SAMP_LEN = len(X[0])
            X = normalize(X, norm='max')
            X = np.reshape(X, (int(NUM_INSTANCES/chan_dic[f]), chan_dic[f], SAMP_LEN))
            y = np.genfromtxt('src/data/processed_datasets/'+f+'_'+l+'.csv', delimiter=',', dtype=int)
            y = to_categorical(y)
            X_test = np.genfromtxt('src/data/processed_datasets/'+f+'_attributes_test.csv', delimiter=',')
            TEST_INSTANCES = len(X_test)
            X_test = normalize(X_test, norm='max')
            X_test = np.reshape(X_test, (int(TEST_INSTANCES/chan_dic[f]), chan_dic[f], SAMP_LEN))
            y_test = np.genfromtxt('src/data/processed_datasets/'+f+'_labels_test.csv', delimiter=',', dtype=int)
            y_test = to_categorical(y_test)
            X, y,  = shuffle(X, y, random_state=1899)
            X_test, y_test = shuffle(X_test, y_test, random_state=1899)
            model = build_cnn(X, class_dic[f], num_channels=chan_dic[f], opt='adam', loss='categorical_crossentropy')
            model = train_cnn(model, X, y)
            score = evaluate_cnn(model, X_test, y_test)
            print("Score for this model: \n", score)
            results_file.write(score)
            results_file.write('\n\n')
            counter += 1
            gc.collect()
            results_file.flush()
    results_file.close()
