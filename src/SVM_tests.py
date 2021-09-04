#Author: Gentry Atkinson
#Organization: Texas University
#Data: 2 September, 2021
#Train and test an SVM on the 6 datasets with their many label sets

import numpy as np
import tensorflow.keras.metrics as met
from sklearn.metrics import classification_report
from sklearn.preprocessing import normalize
from sklearn import svm
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
import gc
import os
from utils.ts_feature_toolkit import calc_AER, calc_TER, get_features_for_set
from datetime import date

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

def build_svm(X, num_classes):
    print("Input Shape: ", X.shape)
    model = svm.SVC()
    return model

def train_svm(model, X, y):
    model.fit(X, y)
    return model

def evaluate_svm(model, X, y_true, mlr):
    y_pred = model.predict(X)
    print('Shape of y true: '.format(y_true))
    print('Shape of y predicted: '.format(y_pred))
    aer = calc_AER(y_true, y_pred)
    ter = calc_TER(aer, mlr)
    return classification_report(y_true, y_pred), confusion_matrix(y_true, y_pred), aer, ter

def absChannels(X, num_channels):
    X_avg = np.zeros((len(X)//num_channels, len(X[0])))
    for i in range(0, len(X_avg)):
        X_avg[i, :] = np.linalg.norm(X[num_channels*i:num_channels*i+num_channels, :], axis=0)
    return X_avg

if __name__ == "__main__":
    if __name__ == "__main__":
        print("Testing SVM")
        print(date.today())
        results_file = open('results/SVM_results.txt', 'w+')
        counter = 1

        for f in sets:
            #matrix of true and apparent error rates
            aer_mat = np.zeros((7, 7))
            ter_mat = np.zeros((7, 7))
            #load the attributes for a test dataset
            X_test = np.genfromtxt('src/data/processed_datasets/'+f+'_attributes_test.csv', delimiter=',')
            X_test = get_features_for_set(X_test)
            X_test = normalize(X_test, norm='max')
            X_test = absChannels(X_test, chan_dic[f])
            TEST_INSTANCES = len(X_test)
            SAMP_LEN = len(X_test[0])
            X_train_feat = np.genfromtxt('src/data/processed_datasets/'+f+'_attributes_train.csv', delimiter=',')
            X_train_feat = get_features_for_set(X_train_feat)
            X_train_feat = normalize(X_train_feat, norm='max')
            X_train_feat = absChannels(X_train_feat, chan_dic[f])
            for i, l_train in enumerate(labels):
                if '5' in l_train:
                    mlr_train = 0.05
                elif '10' in l_train:
                    mlr_train = 0.1
                else:
                    mlr_train = 0.
                #load the training label and attribute sets
                X_train = np.copy(X_train_feat)
                NUM_INSTANCES = len(X_train)
                y_train = np.genfromtxt('src/data/processed_datasets/'+f+'_labels_'+l_train+'.csv', delimiter=',', dtype=int)
                X_train, y_train,  = shuffle(X_train, y_train, random_state=1899)
                model = build_svm(X_train, class_dic[f])
                model = train_svm(model, X_train, y_train)
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
                    print("Shape of X_train: ", X_train.shape)
                    print("Shape of X_test: ", X_test.shape)
                    print("Shape of y_train: ", y_train.shape)
                    print("Shape of y_test: ", y_test.shape)
                    print("NUM_INSTANCES is ", NUM_INSTANCES)
                    print("instances should be ", NUM_INSTANCES//chan_dic[f])
                    score, mat, aer, ter = evaluate_svm(model, X_test, y_test, mlr_test)
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
