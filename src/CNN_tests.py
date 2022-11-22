#Author: Gentry Atkinson
#Organization: Texas State University
#Data: 21 July, 2021
#Train and test a CNN on the 6 datasets with their many label sets

#TODO:
#  -Test noise levels from 1-30%
#  -Test clean train labels too!
#  -Tune up the CNN (consider pytorch)
#  -Write the results as Pandas
#  -Change the metrics

import numpy as np
from tensorflow import keras
#from keras import Sequential
from tensorflow.keras import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Input
from tensorflow.keras.layers import Reshape, BatchNormalization, Dropout, ReLU, GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.losses import CategoricalCrossentropy
# import torch
# from torch import nn
from sklearn.metrics import classification_report
from sklearn.preprocessing import normalize
import gc
import os
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from utils.ts_feature_toolkit import calc_AER, calc_TER, calc_bias_metrics, calc_error_rates
from datetime import date
from model_config import loadDic

# device = "cuda" if torch.cuda.is_available() else "cpu"

DEBUG = True
SMOOTHING_RATE = 0.0

if DEBUG:
    sets = [
        'ss2', 'ss1'
    ]
else:
    sets = [
        'ss1', 'ss2', 'har1', 'har2', 'sn1', 'sn2'
    ]

labels = [
    'ncar', 'nar', 'nnar'
]

chan_dic = {
    'har1':1, 'har2':3, 'ss1':1, 'ss2':1, 'sn1':6, 'sn2':5
}

class_dic = {
    'har1':6, 'har2':6, 'ss1':2, 'ss2':5, 'sn1':2, 'sn2':5
}

FPR = 0
FNR = 0

config_dic = loadDic('CNN')
"""
Build CNN
Construct and return a compiled CNN
Parameters:
    X, numpy_array, trainging data
    num_classes, int,  number of classes present in label set
    set, string, identifier of dataset X was read from
    num_channels, int, the number of channel in X  
"""

def build_cnn(
    X : np.ndarray, 
    num_classes : int, 
    set : str, 
    num_channels=1, 
    opt='SGD', 
    loss='mean_squared_error'
):
    print("Input Shape: ", X.shape)
    model = Sequential([
        Input(shape=X[0].shape),
        BatchNormalization(),
        Conv1D(filters=config_dic[set]['l1_numFilters']*1, kernel_size=config_dic[set]['l1_kernelSize'], padding='causal', activation='relu', groups=1),
        Conv1D(filters=config_dic[set]['l1_numFilters']*1, kernel_size=config_dic[set]['l1_kernelSize'], padding='causal', activation='relu', groups=1),
        MaxPooling1D(pool_size=(config_dic[set]['l1_maxPoolSize']*1), data_format='channels_first'),
        Conv1D(filters=config_dic[set]['l2_numFilters'], kernel_size=config_dic[set]['l2_kernelSize'], padding='causal', activation='relu', groups=1),
        Conv1D(filters=config_dic[set]['l2_numFilters'], kernel_size=config_dic[set]['l2_kernelSize'], padding='causal', activation='relu', groups=1),
        MaxPooling1D(pool_size=(config_dic[set]['l2_maxPoolSize']), data_format='channels_first'),
        Dropout(config_dic[set]['dropout']),
        GlobalAveragePooling1D(data_format="channels_first"),
        Dense(num_classes, activation='softmax')
    ])
    # model = nn.Sequential(
    #     nn.Conv1d(in_channels=num_channels, out_channels=config_dic[set]['l1_numFilters'], 
    #         kernel_size=config_dic[set]['l1_kernelSize'], padding='zeros'
    #     ),
    #     nn.Conv1d(in_channels=config_dic[set]['l1_numFilters'], out_channels=config_dic[set]['l1_numFilters'],
    #         kernel_size=config_dic[set]['l1_kernelSize'], padding='zeros'
    #     ),
    #     nn.BatchNorm1d(config_dic[set]['l1_numFilters']),
    #     nn.ReLU(),
    #     nn.MaxPool1d(pool_size=config_dic[set]['l1_maxPoolSize']),
    #     nn.Dropout(0.25),
    #     nn.LazyConv1d(out_channels=config_dic[set]['l2_numFilters'], 
    #         kernel_size=config_dic[set]['l1_kernelSize'], padding='zeros'
    #     ),
    #     nn.Conv1d(in_channels=config_dic[set]['l2_numFilters'],out_channels=config_dic[set]['l2_numFilters'],
    #         kernel_size=config_dic[set]['l2_kernelSize'], padding='zeros'
    #     ),
    #     nn.BatchNorm1d(config_dic[set]['l2_numFilters']),
    #     nn.ReLU(),
    #     nn.MaxPool1d(pool_size=config_dic[set]['l2_maxPoolSize']),
    #     nn.Dropout(0.25),
    #     nn.AdaptiveAvgPool1d(output_size=num_classes*4),
    #     nn.Linear(in_features=num_classes*4, out_features=num_classes),
    #     nn.Softmax()
    # )
    # model = model.to(device)
    #print(model)
    print(model.summary())
    return model

def train_cnn(model, X, y):
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)
    rlr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001)
    NUM_CORES = os.cpu_count()
    print('Size of X to fit: ', X.shape)
    model.fit(X, y, epochs=500, verbose=0, callbacks=[es, rlr], validation_split=0.1, batch_size=32, use_multiprocessing=True, workers=NUM_CORES)
    return model

def evaluate_cnn(model, X, y):
    y_pred = model.predict(X)
    y_pred = np.argmax(y_pred, axis=-1)
    y_true = np.argmax(y, axis=-1)
    print('Shape of y true: {}'.format(y_true.shape))
    print('Shape of y predicted: {}'.format(y_pred.shape))
    aer = calc_AER(y_true, y_pred)
    print(base_fpr, base_fnr)
    return classification_report(y_true, y_pred), confusion_matrix(y_true, y_pred), aer


if __name__ == "__main__":
    print("Testing CNN")
    print(date.today())
    results_file = open('results/CNN_results.txt', 'w+')
    results_file.write('{}\n'.format(date.today()))
    readable_file = open('results/all_results.txt', 'w+')
    readable_file.write('{}\n'.format(date.today()))
    readable_file.write('######  CNN #####\n')

    counter = 1

    for data_set in sets:
        #matrix of true and apparent error rates
        aer_dict = {}
        ter_dict = {}
        #matrix of bias measures
        cev_mat = np.zeros((7, 7))
        sde_mat = np.zeros((7, 7))
        #load the attributes for a test dataset
        X_test = np.load('src/data/processed_datasets/'+data_set+'_attributes_test.npy')
        X_test = normalize(X_test, norm='max')
        print('Shape of X_test: '. X_test.shape)
        TEST_INSTANCES = len(X_test)
        SAMP_LEN = len(X_test[0])
        #X_test = np.reshape(X_test, (int(TEST_INSTANCES//chan_dic[f]), chan_dic[f], SAMP_LEN))
        base_fpr = None
        base_fnr = None
        for i, noise_type in enumerate(labels):
            for mlr in range(0, 30):
                mlr_percent = mlr/100
                #load the training label and attribute sets
                X_train = np.load('src/data/processed_datasets/'+data_set+'_attributes_train.npy')
                X_train = normalize(X_train, norm='max')
                NUM_INSTANCES = len(X_train)
                #X_train = np.reshape(X_train, (int(NUM_INSTANCES//chan_dic[f]), chan_dic[f], SAMP_LEN))
                y_train = np.load('src/data/processed_datasets/'+data_set+'_labels_'+noise_type+str(mlr)+'.npy')
                y_train = to_categorical(y_train)
                X_train, y_train,  = shuffle(X_train, y_train, random_state=1899)
                model = build_cnn(X_train, class_dic[data_set], set=data_set, num_channels=chan_dic[data_set], opt='adam', loss=CategoricalCrossentropy(label_smoothing=SMOOTHING_RATE))
                model = train_cnn(model, X_train, y_train)

                #Test on noisy labels
                print ('Experiment: ', counter, " Set: ", data_set, "Train Labels: ", noise_type+str(mlr), "Test Labels: ", noise_type+str(mlr))
                results_file.write('############Experiment {}############\n'.format(counter))
                results_file.write('Set: {}\n'.format(data_set))
                results_file.write('Train Labels: {}{}\n'.format(noise_type, mlr))
                results_file.write('Test Labels: {}{}\n'.format(noise_type, mlr))
                #load the test attribute set
                y_test = np.load('src/data/processed_datasets/'+data_set+'_labels_test_'+noise_type+str(mlr)+'.npy')
                y_test = to_categorical(y_test)
                print("Shape of X_train: ", X_train.shape)
                print("Shape of X_test: ", X_test.shape)
                print("Shape of y_train: ", y_train.shape)
                print("Shape of y_test: ", y_test.shape)
                print("NUM_INSTANCES is ", NUM_INSTANCES)
                print("instances should be ", NUM_INSTANCES//chan_dic[data_set])
                score, mat, aer = evaluate_cnn(model, X_test, y_test)
                aer_dict[data_set + noise_type + str(mlr)] = aer
                print("Score for this model: \n", score)
                print("Confusion Matrix for this model: \n", mat)
                print("Apparent error rate: \n", aer)
                results_file.write(score)
                results_file.write('\nColumns are predictions, rows are labels\n')
                results_file.write(str(mat))
                results_file.write('\n')
                results_file.write(f'Apparent error rate: {aer}')
                counter += 1
                gc.collect()
                results_file.flush()

                #Test on clean labels
                print ('Experiment: ', counter, " Set: ", data_set, "Train Labels: ", noise_type+str(mlr), "Test Labels: Clean")
                results_file.write('############Experiment {}############\n'.format(counter))
                results_file.write('Set: {}\n'.format(data_set))
                results_file.write('Train Labels: {}{}\n'.format(noise_type, mlr))
                results_file.write('Test Labels: {}{}\n'.format(noise_type, mlr))
                #load the test attribute set
                y_test = np.load('src/data/processed_datasets/'+data_set+'_labels_test_clean.npy')
                y_test = to_categorical(y_test)
                print("Shape of X_train: ", X_train.shape)
                print("Shape of X_test: ", X_test.shape)
                print("Shape of y_train: ", y_train.shape)
                print("Shape of y_test: ", y_test.shape)
                print("NUM_INSTANCES is ", NUM_INSTANCES)
                print("instances should be ", NUM_INSTANCES//chan_dic[data_set])
                score, mat, ter = evaluate_cnn(model, X_test, y_test)
                ter_dict[data_set + noise_type + str(mlr)] = ter
                print("Score for this model: \n", score)
                print("Confusion Matrix for this model: \n", mat)
                print("True error rate: \n", ter)
                results_file.write(score)
                results_file.write('\nColumns are predictions, rows are labels\n')
                results_file.write(str(mat))
                results_file.write('\n')
                results_file.write(f'True error rate: {ter}')
                counter += 1
                gc.collect()
                results_file.flush()



        results_file.write("Summary of {}\n".format(data_set))
        readable_file.write("Summary of {}\n".format(data_set))
        results_file.write('Apparent Error Rates.\n')
        readable_file.write('Apparent Error Rates.\n')
        results_file.write('Label Sets: {}\n'.format(labels))
        results_file.write(str(aer_dict))
        
        
    results_file.close()
    readable_file.close()
