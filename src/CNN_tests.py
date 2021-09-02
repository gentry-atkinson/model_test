#Author: Gentry Atkinson
#Organization: Texas University
#Data: 21 July, 2021
#Train and test a CNN on the 6 datasets with their many label sets

import numpy as np
import tensorflow.keras.metrics as met
from tensorflow.keras import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Input
from tensorflow.keras.layers import Reshape, BatchNormalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report
from sklearn.preprocessing import normalize
import gc
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from utils.ts_feature_toolkit import calc_AER, calc_TER

DEBUG = False

if DEBUG:
    sets = [
        'ss1', 'ss2'
    ]
else:
    sets = [
        'ss1', 'ss2', 'bs1', 'bs2', 'har1', 'har2'
    ]

labels = [
    'clean', 'ncar5', 'ncar10', 'nar5',
    'nar10', 'nnar5', 'nnar10'
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
        #Conv1D(filters=128, kernel_size=16, activation='relu', padding='same'),
        MaxPooling1D(pool_size=(2), data_format='channels_first'),
        Dropout(0.25),
        #Conv1D(filters=64, kernel_size=32, activation='relu', padding='same'),
        Conv1D(filters=64, kernel_size=16, activation='relu', padding='same'),
        MaxPooling1D(pool_size=(2), data_format='channels_first'),
        Dropout(0.25),
        Flatten(),
        Dense(128, activation='relu'),
        #Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=opt, loss=loss, metrics=[met.CategoricalAccuracy()])
    model.summary()
    return model

def train_cnn(model, X, y):
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=7)
    NUM_CORES = os.cpu_count()
    model.fit(X, y, epochs=100, verbose=1, callbacks=[es], validation_split=0.1, batch_size=100, workers=NUM_CORES)
    return model

def evaluate_cnn(model, X, y, mlr):
    y_pred = model.predict(X)
    y_pred = np.argmax(y_pred, axis=-1)
    y_true = np.argmax(y, axis=-1)
    print('Shape of y true: '.format(y_true))
    print('Shape of y predicted: '.format(y_pred))
    aer = calc_AER(y_true, y_pred)
    ter = calc_TER(aer, mlr)
    return classification_report(y_true, y_pred), confusion_matrix(y_true, y_pred), aer, ter

if __name__ == "__main__":
    print("Testing CNN")
    results_file = open('results/CNN_results.txt', 'w+')
    counter = 1

    for f in sets:
        #matrix of true and apparent error rates
        aer_mat = np.zeros((7, 7))
        ter_mat = np.zeros((7, 7))
        #load the attributes for a test dataset
        X_test = np.genfromtxt('src/data/processed_datasets/'+f+'_attributes_test.csv', delimiter=',')
        X_test = normalize(X_test, norm='max')
        TEST_INSTANCES = len(X_test)
        SAMP_LEN = len(X_test[0])
        X_test = np.reshape(X_test, (int(TEST_INSTANCES//chan_dic[f]), chan_dic[f], SAMP_LEN))
        for i, l_train in enumerate(labels):
            if '5' in l_train:
                mlr_train = 0.05
            elif '10' in l_train:
                mlr_train = 0.1
            else:
                mlr_train = 0.
            #load the training label and attribute sets
            X_train = np.genfromtxt('src/data/processed_datasets/'+f+'_attributes_train.csv', delimiter=',')
            X_train = normalize(X_train, norm='max')
            NUM_INSTANCES = len(X_train)
            X_train = np.reshape(X_train, (int(NUM_INSTANCES//chan_dic[f]), chan_dic[f], SAMP_LEN))
            y_train = np.genfromtxt('src/data/processed_datasets/'+f+'_labels_'+l_train+'.csv', delimiter=',', dtype=int)
            y_train = to_categorical(y_train)
            X_train, y_train,  = shuffle(X_train, y_train, random_state=1899)
            model = build_cnn(X_train, class_dic[f], num_channels=chan_dic[f], opt='adam', loss='categorical_crossentropy')
            model = train_cnn(model, X_train, y_train)
            for j, l_test in enumerate(labels):
                if '5' in l_test:
                    mlr_test = 0.05
                elif '10' in l_test:
                    mlr_test = 0.1
                else:
                    mlr_test = 0.
                print ('Experiment: ', counter, " Set: ", f, "Train Labels: ", l_train, "Test Labels: ", l_test)
                results_file.write('############Experiment {}############\n'.format(counter))
                results_file.write('Set: {}\n'.format(f))
                results_file.write('Train Labels: {}\n'.format(l_train))
                results_file.write('Test Labels: {}\n'.format(l_test))
                #load the test attribute set
                y_test = np.genfromtxt('src/data/processed_datasets/'+f+'_labels_test_'+l_test+'.csv', delimiter=',', dtype=int)
                y_test = to_categorical(y_test)
                print("Shape of X_train: ", X_train.shape)
                print("Shape of X_test: ", X_test.shape)
                print("Shape of y_train: ", y_train.shape)
                print("Shape of y_test: ", y_test.shape)
                print("NUM_INSTANCES is ", NUM_INSTANCES)
                print("instances should be ", NUM_INSTANCES//chan_dic[f])
                score, mat, aer, ter = evaluate_cnn(model, X_test, y_test, mlr_test)
                aer_mat[i, j] = aer
                ter_mat[i, j] = ter
                print("Score for this model: \n", score)
                print("Confusion Matrix for this model: \n", mat)
                results_file.write(score)
                results_file.write('\nColumns are predictions, rows are labels\n')
                results_file.write(str(mat))
                results_file.write('\n')
                results_file.write('AER: {:.3f} MLR_train: {} MLR_test:{} TER: {:.3f}'.format(aer, mlr_train, mlr_test, ter))
                results_file.write('\n\n')
                counter += 1
                gc.collect()
                results_file.flush()
        results_file.write("Summary of {}\n".format(f))
        results_file.write('Apparent Error Rates. Row->Train Column->Test\n')
        results_file.write('Label Sets: {}\n'.format(labels))
        for row in aer_mat:
            for item in row:
                results_file.write('{:.3f}\t'.format(item))
            results_file.write('\n')
        results_file.write('\n\nTrue Error Rates. Row->Train Column->Test\n')
        results_file.write('Label Sets: {}\n'.format(labels))
        for row in ter_mat:
            for item in row:
                results_file.write('{:.3f}\t'.format(item))
            results_file.write('\n')
        results_file.write('\n\n')
        results_file.flush()
    results_file.close()
