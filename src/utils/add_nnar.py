#Author: Gentry Atkinson
#Organization: Texas University
#Data: 12 May, 2021
#Generate two Noise Not at Random label sets, 5% and 10%

#TODO:
#  -Use the output of KNN to pick the noisy label

import numpy as np
from random import randint
import os
from sklearn.neighbors import NearestNeighbors
from utils.ts_feature_toolkit import get_features_for_set

def absChannels(X, num_channels):
    print("Length of attribute list in add_nnar: ", len(X))
    print("Length of signal in nnar: ", len(X[0]))
    print("Number of channels in nnar: ", num_channels)
    X_avg = np.zeros((len(X)//num_channels, len(X[0])))
    for i in range(0, len(X_avg)):
        X_avg[i, :] = np.linalg.norm(X[num_channels*i:num_channels*i+num_channels, :], axis=0)
    return X_avg

"""
Noise Not at Random-> the mislabeling rate is affected by class and features

Find instances of the majority class whose nearest neighbor in the extracted feature
space is not from the majority class. Relabel those instances as being from the minor
class. Other instances of the major class will have a uniform 3% mislabeling rate.
"""

def add_nnar(
    attributes : np.ndarray, 
    clean_labels : np.ndarray, 
    filename : str, 
    num_classes : int,
    mislab_rate: int, 
    num_channels=1, 
    att_file="",
):
    """
    Write a noisy label file with a specific mislabeling rate (0-100)
    """
    if attributes is None:
        attributes = np.load(att_file)
    
    total_flipped = 0
    total_counter = 0

    counts = [np.count_nonzero(clean_labels==i) for i in range(num_classes)]
    MAJ_LABEL = int(np.argmax(counts))
    MIN_LABEL = int(np.argmin(counts))
    SET_LENGTH = len(clean_labels)
    assert MAJ_LABEL != MIN_LABEL, "Calculating class imbalance has gone horribly wrong"

    attributes = get_features_for_set(attributes, len(attributes))
    noisy_labels = clean_labels.copy()

    if os.path.exists(f'{filename}_knn_output.npy'):
        i = np.load(f'{filename}_knn_output.npy')
    else:
        nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(attributes)
        d, i = nbrs.kneighbors(attributes)
        np.save(f'{filename}_knn_output.npy', i)

    while total_flipped < (mislab_rate/100)*SET_LENGTH:
        rand_instance_index = randint(0, SET_LENGTH-1)
        total_counter += 1
        if noisy_labels[rand_instance_index] == MAJ_LABEL:
            if noisy_labels[i[rand_instance_index][1]]!=MAJ_LABEL or randint(0,99)<=(mislab_rate/10):
                #noisy_labels[rand_instance_index] = MIN_LABEL
                noisy_labels[rand_instance_index] = i[rand_instance_index][1]
                total_flipped += 1

    np.save(f'{filename}_nnar_{mislab_rate}.npy', noisy_labels)

    #Sanity checks
    print('NNAR')
    print('Len of clean labels: ', len(clean_labels))
    print('Len of noisy labels: ', len(noisy_labels))
    print('Mislabeling rate: ', mislab_rate)
    print('Number of noisy labels: ', total_flipped)
    print('Total iterations: ', total_counter)
    print('Number of clean labels: ', np.count_nonzero(noisy_labels==clean_labels))

if __name__ == '__main__':
    att = np.concatenate((np.zeros((2500, 3, 50), dtype=int), np.ones((2000, 3, 50), dtype=int)), axis=0)
    clean_labels = np.concatenate((np.zeros((2500), dtype=int), np.ones((2000), dtype=int)), axis=0)
    add_nnar(att, clean_labels, 'test_labels', 2, 10)
    noisy_labels = np.load('test_labels_nnar_10.npy')
    
