#Author: Gentry Atkinson and Alex Katrompas
#Organization: Texas State University
#Data: 21 January, 2022
#Train and test a 1D transformer on the 6 datasets with their many label sets

#original transformer code at:  https://github.com/imics-lab/recurrence-with-self-attention

#TODO:
#  -Test noise levels from 1-30%
#  -Tune up the Transformer (consider pytorch)
#  -Write the results as Pandas
#  -Change the metrics

import numpy as np
import tensorflow.keras.metrics as met
from tensorflow.keras import Sequential, Model
from tensorflow.keras.utils import to_categorical
import tensorflow.keras.layers as layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report
from sklearn.preprocessing import normalize
import gc
import os
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from utils.ts_feature_toolkit import calc_AER, calc_TER, calc_bias_metrics, calc_error_rates
from model_config import loadDic
from datetime import date

DEBUG = False

if DEBUG:
    sets = [
        'har2', 'har1'
    ]
else:
    sets = [
        'ss1', 'ss2', 'har1', 'har2', 'sn1', 'sn2'
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
    'har1':1, 'har2':3, 'ss1':1, 'ss2':1, 'sn1':6, 'sn2':5
}

class_dic = {
    'har1':6, 'har2':6, 'ss1':2, 'ss2':5, 'sn1':2, 'sn2':5
}

FPR = 0
FNR = 0

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    """
    TransformerEncoder for time-series encoder, conforming to the
    transformer architecture from Attention Is All You Need (Vaswani 2017)
    https://arxiv.org/abs/1706.03762
    @param (int) num_heads: Number of Attention Heads
    @param (int) head_size : Head Size
    @param (int) ff_dim: Feed Forward Dimension
    @param (float) dropout : Dropout (between 0 and .99)
    
    Return: transformer encoder
    """

    # Normalization and Attention
    print(type(inputs))
    x = layers.BatchNormalization()(inputs)
    x = layers.MultiHeadAttention(num_heads=num_heads, key_dim=head_size, dropout=dropout)(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation='relu')(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res 

config_dic = loadDic('Transformer')

def build_tran(X, num_classes, set, opt='SGD', loss='mean_squared_error'):
    print("Input Shape: ", X.shape)
    inputs = layers.Input(shape=X[0].shape)
    x = inputs
    for _ in range(config_dic[set]['num_attn+layers']):
        x = transformer_encoder(x, config_dic[set]['head_size'], config_dic[set]['num_heads'], config_dic[set]['ff_dim'], config_dic[set]['dropout'])
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Flatten()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = Model(inputs, outputs)
    model.compile(optimizer=opt, loss=loss, metrics=[met.CategoricalAccuracy()])
    model.summary()
    return model

def train_tran(model, X, y):
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)
    rlr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001)
    NUM_CORES = os.cpu_count()
    model.fit(X, y, epochs=500, verbose=0, callbacks=[es, rlr], validation_split=0.1, batch_size=32, workers=NUM_CORES)
    return model

def evaluate_tran(model, X, y, mlr, base_fpr, base_fnr):
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


if __name__ == "__main__":
    print("Testing Transformer")
    print(date.today())
    results_file = open('results/Tran_results.txt', 'w+')
    results_file.write('{}\n'.format(date.today()))
    readable_file = open('results/all_results.txt', 'a')
    readable_file.write('{}\n'.format(date.today()))
    readable_file.write('######  Tran #####\n')

    counter = 1

    for f in sets:
        #matrix of true and apparent error rates
        aer_mat = np.zeros((7, 7))
        ter_mat = [
            ["","","","","","",""],
            ["","","","","","",""],
            ["","","","","","",""],
            ["","","","","","",""],
            ["","","","","","",""],
            ["","","","","","",""],
            ["","","","","","",""]
        ]
        #matrix of bias measures
        cev_mat = np.zeros((7, 7))
        sde_mat = np.zeros((7, 7))
        #load the attributes for a test dataset
        X_test = np.genfromtxt('src/data/processed_datasets/'+f+'_attributes_test.csv', delimiter=',')
        X_test = normalize(X_test, norm='max')
        TEST_INSTANCES = len(X_test)
        SAMP_LEN = len(X_test[0])
        X_test = np.reshape(X_test, (int(TEST_INSTANCES//chan_dic[f]), chan_dic[f], SAMP_LEN))
        base_fpr = None
        base_fnr = None
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
            model = build_tran(X_train, class_dic[f], set=f, opt='adam', loss='categorical_crossentropy')
            model = train_tran(model, X_train, y_train)
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
                score, mat, aer, ter, cev, sde = evaluate_tran(model, X_test, y_test, mlr_test, base_fpr, base_fnr)
                if i==0 and j==0:
                    FP = mat.sum(axis=0) - np.diag(mat)
                    FN = mat.sum(axis=1) - np.diag(mat)
                    TP = np.diag(mat)
                    TN = mat.sum() - (FP + FN + TP)
                    base_fpr = FP / (FP  + TN)
                    base_fnr = FN / (FN + TP)
                print("Recording results in matrix at {} {}".format(i, j))
                print("AER: ", aer)
                print("TER: ", ter)
                aer_mat[i, j] = aer
                ter_mat[i][j] = ter
                cev_mat[i, j] = cev
                sde_mat[i, j] = sde
                print("Score for this model: \n", score)
                print("Confusion Matrix for this model: \n", mat)
                results_file.write(score)
                results_file.write('\nColumns are predictions, rows are labels\n')
                results_file.write(str(mat))
                results_file.write('\n')
                results_file.write('AER: {:.3f} MLR_train: {} MLR_test:{} TER: {}'.format(aer, mlr_train, mlr_test, ter))
                results_file.write('\n\n')
                counter += 1
                gc.collect()
                results_file.flush()
        results_file.write("Summary of {}\n".format(f))
        readable_file.write("Summary of {}\n".format(f))
        results_file.write('Apparent Error Rates. Row->Train Column->Test\n')
        readable_file.write('Apparent Error Rates. Row->Train Column->Test\n')
        results_file.write('Label Sets: {}\n'.format(labels))
        for row in aer_mat:
            for item in row:
                results_file.write('{:.3f}\t'.format(item))
                readable_file.write('{:.3f}\t'.format(item))
            results_file.write('\n')
            readable_file.write('\n')
        results_file.write('\n\nTrue Error Rates. Row->Train Column->Test\n')
        readable_file.write('\n\nTrue Error Rates. Row->Train Column->Test\n')
        results_file.write('Label Sets: {}\n'.format(labels))
        for row in ter_mat:
            for item in row:
                results_file.write('{}\t'.format(item))
                readable_file.write('{}\t'.format(item))
            results_file.write('\n')
            readable_file.write('\n')
        results_file.write('\n\nCEV. Row->Train Column->Test\n')
        readable_file.write('\n\nCEV. Row->Train Column->Test\n')
        results_file.write('Label Sets: {}\n'.format(labels))
        for row in cev_mat:
            for item in row:
                results_file.write('{:.3f}\t'.format(item))
                readable_file.write('{:.3f}\t'.format(item))
            results_file.write('\n')
            readable_file.write('\n')
        results_file.write('\n\n')
        results_file.write('\n\nSDE. Row->Train Column->Test\n')
        readable_file.write('\n\nSDE. Row->Train Column->Test\n')
        results_file.write('Label Sets: {}\n'.format(labels))
        for row in sde_mat:
            for item in row:
                results_file.write('{:.3f}\t'.format(item))
                readable_file.write('{:.3f}\t'.format(item))
            results_file.write('\n')
            readable_file.write('\n')
        results_file.write('\n\n')
        results_file.flush()
        readable_file.flush()
    results_file.close()
    readable_file.close()
