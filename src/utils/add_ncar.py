#Author: Gentry Atkinson
#Organization: Texas University
#Data: 11 May, 2021
#Generate two Noise Completely at Random label sets, 5% and 10%

import numpy as np
from random import randint
import os

def new_label_oneHot(old_label: int , num_classes: int) -> int:
    n = old_label
    while(n==old_label):
        n = randint(0, num_classes-1)
    return n

def new_label_vector(old_label: np.ndarray, num_classes: int) -> np.ndarray:
    return np.roll(old_label, randint(1, num_classes-1))

"""
Noise Completely at Random-> the mislabeling rate is independent of class and features

Apply a flat mislabeling rate to all instances in a dataset. The low noise label set
will have a uniform 5% mislabeling rate. The high noise label set will have a uniform
10% mislabel rate.
"""

def add_ncar(
    clean_labels : np.ndarray, 
    filename: str, 
    num_classes: int,
    mislab_rate: int
) -> None:
    """
    Write a noisy label file with a specific mislabeling rate (0-100)
    """
    if clean_labels.ndim == 1:
        new_label = new_label_oneHot
        noisy_labels = np.empty((len(clean_labels)))
    elif clean_labels.ndim == 2:
        new_label = new_label_vector
        noisy_labels = np.empty((len(clean_labels), num_classes))
    else:
        print('Unusual labels passed to add_ncar')
        return
    
    flipped_counter = 0
    for i, l in enumerate(clean_labels):
        if randint(0, 99) < mislab_rate:
            noisy_labels[i] = new_label(l, num_classes)
            flipped_counter += 1
        else:
            noisy_labels[i] = l

    np.save(f'{filename}_ncar_{mislab_rate}.npy', noisy_labels)

    #Sanity checks
    print('NCAR')
    print('Len of clean labels: ', len(clean_labels))
    print('Len of noisy labels: ', len(noisy_labels))
    print('Mislabeling rate: ', mislab_rate)
    print('Number of noisy labels: ', flipped_counter)
    print('Number of clean labels: ', np.count_nonzero(noisy_labels==clean_labels))

if __name__ == '__main__':
    clean_labels = np.concatenate((np.zeros((25000), dtype=int), np.ones((25000), dtype=int)), axis=0)
    add_ncar(clean_labels, 'test_labels', 2, 10)
    noisy_labels = np.load('test_labels_ncar_10.npy')

