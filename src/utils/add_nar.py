#Author: Gentry Atkinson
#Organization: Texas University
#Data: 12 May, 2021
#Generate two Noise at Random label sets, 5% and 10%

#TODO:
#  -Pass in noise percect for add_nar
#  -Test output

import numpy as np
from random import randint
import os

"""
Noise at Random-> The mislabeling rate is influenced by class

Mislabel only instances from the majority class. The total mislabeling rate of the
low noise label set will be 5% and the total mislabeling rate of the high noise
set will be 10%.
"""
# def add_nar(clean_labels, filename, num_classes):
#     low_noise_labels = open(filename + '_nar5.csv', 'w+')
#     high_noise_labels = open(filename + '_nar10.csv', 'w+')
#     low_indexes = open(filename + '_nar5_indexes.csv', 'w+')
#     high_indexes = open(filename + '_nar10_indexes.csv', 'w+')

#     total_counter = 0
#     l_flipped_counter = 0
#     h_flipped_counter = 0

#     counts = [np.count_nonzero(clean_labels==i) for i in range(num_classes)]
#     print("Label counts in add_nar: ", counts)
#     MAJ_LABEL = int(np.argmax(counts))
#     MIN_LABEL = int(np.argmin(counts))

#     assert MAJ_LABEL != MIN_LABEL, "Calculating class imbalance has gone horribly wrong"

#     imbalance = len(clean_labels)/counts[MAJ_LABEL]

#     assert imbalance < 10, "ERROR: imbalance is to high for NAR"

#     for i,l in enumerate(clean_labels):
#         total_counter += 1
#         if l==MAJ_LABEL and randint(0,100)<5*imbalance:
#             low_noise_labels.write('{}\n'.format(MIN_LABEL))
#             low_indexes.write('{}\n'.format(i))
#             l_flipped_counter += 1
#         else:
#             low_noise_labels.write('{}\n'.format(int(l)))

#         if l==MAJ_LABEL and randint(0,100)<10*imbalance:
#             high_noise_labels.write('{}\n'.format(MIN_LABEL))
#             high_indexes.write('{}\n'.format(i))
#             h_flipped_counter += 1
#         else:
#             high_noise_labels.write('{}\n'.format(int(l)))


#     low_noise_labels.close()
#     high_noise_labels.close()

#     #sanity checks
#     print('---NAR---')
#     print('Major label: ', MAJ_LABEL)
#     print('Minor label: ', MIN_LABEL)
#     print('Class imbalance: ', counts[MAJ_LABEL]/(counts[MIN_LABEL] if counts[MIN_LABEL] != 0 else 1))
#     print('Total labels processed: ', total_counter)
#     print('Low noise labels flipped: ', l_flipped_counter)
#     print('High noise labels flipped: ', h_flipped_counter)
#     print('Lines written to low noise file: ')
#     os.system('cat {} | wc -l'.format(filename + '_nar5.csv'))
#     print('Lines written to high noise file: ')
#     os.system('cat {} | wc -l'.format(filename + '_nar10.csv'))

def add_nar(
    clean_labels : np.ndarray,
    filename : str, 
    num_classes : int, 
    mislab_rate : int
):
    """
    Write a noisy label file with a specific mislabeling rate (0-100)
    """
    total_counter = 0
    flipped_counter = 0

    counts = [np.count_nonzero(clean_labels==i) for i in range(num_classes)]
    print("Label counts in add_nar: ", counts)
    MAJ_LABEL = int(np.argmax(counts))
    MIN_LABEL = int(np.argmin(counts))

    assert MAJ_LABEL != MIN_LABEL, "Calculating class imbalance has gone horribly wrong"

    imbalance = len(clean_labels)/counts[MAJ_LABEL]

    assert imbalance < 10, "ERROR: imbalance is to high for NAR"

    if clean_labels.ndim == 1:
        noisy_labels = np.empty((len(clean_labels)))
        ONE_HOT = False
    elif clean_labels.ndim == 2:
        noisy_labels = np.empty((len(clean_labels), num_classes))
        ONE_HOT = True
    else:
        print('Unusual labels passed to add_ncar')
        return

    for i,l in enumerate(clean_labels):
        total_counter += 1
        if l==MAJ_LABEL and randint(0,100)<mislab_rate*imbalance:
            noisy_labels[i] = MIN_LABEL if not ONE_HOT else [1 if i == MIN_LABEL else 0 for i in range(len(l))]
            flipped_counter += 1
        else:
            noisy_labels[i] = l

    #Sanity checks
    print('Len of clean labels: ', len(clean_labels))
    print('Len of noisy labels: ', len(noisy_labels))
    print('Mislabeling rate: ', mislab_rate)
    print('Number of noisy labels: ', flipped_counter)
    print('Number of clean labels: ', np.count_nonzero(noisy_labels==clean_labels)) 


if __name__ == '__main__':
    clean_labels = np.concatenate((np.zeros((25000), dtype=int), np.ones((20000), dtype=int)), axis=0)
    add_nar(clean_labels, 'test_labels', 2, 10)
    noisy_labels = np.load('test_labels_ncar_10.npy')

