#Author: Gentry Atkinson
#Organization: Texas State University
#Data: 1 Dec, 2021
#Process transformer results

import numpy as np
from math import floor
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

def read_file(filename):
    file = open(filename, 'r')
    #Dear lord forgive me for this line of code
    results = [list(map(float, i.split(','))) for i in  file.read().split('\n') if i != '']
    true_labels = np.array([floor(i[0]) for i in results], dtype='int')
    pred_labels = np.array([floor(i[1]) for i in results], dtype='int')
    return true_labels, pred_labels


if __name__ == '__main__':
    sets = ['har1', 'ss1', 'ss2', 'bs1', 'bs2']
    noise = ['clean', 'ncar5', 'ncar10', 'nar5', 'nar10', 'nnar5', 'nnar10']

    print('### Reading All Transformer Results ###\n')
    for s in sets:
        for n in noise:
            print ('Set: '+s, ' Noise Type: '+n)
            y_true, y_pred = read_file('results/transformer/'+s+'_'+n+'_results.csv')
            mat = confusion_matrix(y_true, y_pred)
            print(mat)
            rep = classification_report(y_true, y_pred)
            print(rep)
