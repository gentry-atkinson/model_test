#Author: Gentry Atkinson
#Organization: Texas University
#Data: 12 May, 2021
#Generate two Noise at Random label sets, 5% and 10%

import numpy as np
from random import randint
import os

"""
Noise at Random-> The mislabeling rate is influenced by class

Mislabel only instances from the majority class. The total mislabeling rate of the
low noise label set will be 5% and the total mislabeling rate of the high noise
set will be 10%.
"""
def add_nar(clean_labels, filename, num_classes):
    low_noise_labels = open(filename + '_nar5.csv', 'w+')
    high_noise_labels = open(filename + '_nar10.csv', 'w+')
    low_indexes = open(filename + '_nar5_indexes.csv', 'w+')
    high_indexes = open(filename + '_nar10_indexes.csv', 'w+')

    total_counter = 0
    l_flipped_counter = 0
    h_flipped_counter = 0

    counts = [np.count_nonzero(clean_labels==i) for i in range(num_classes)]
    MAJ_LABEL = int(np.argmax(counts))
    MIN_LABEL = int(np.argmin(counts))

    assert MAJ_LABEL != MIN_LABEL, "Calculating class imbalance has gone horribly wrong"

    imbalance = len(clean_labels)/counts[MAJ_LABEL]

    assert imbalance < 10, "ERROR: imbalance is to high for NAR"

    for i,l in enumerate(clean_labels):
        total_counter += 1
        if l==MAJ_LABEL and randint(0,100)<5*imbalance:
            low_noise_labels.write('{}\n'.format(MIN_LABEL))
            low_indexes.write('{}\n'.format(i))
            l_flipped_counter += 1
        else:
            low_noise_labels.write('{}\n'.format(int(l)))

        if l==MAJ_LABEL and randint(0,100)<10*imbalance:
            high_noise_labels.write('{}\n'.format(MIN_LABEL))
            high_indexes.write('{}\n'.format(i))
            h_flipped_counter += 1
        else:
            high_noise_labels.write('{}\n'.format(int(l)))


    low_noise_labels.close()
    high_noise_labels.close()

    #sanity checks
    print('---NAR---')
    print('Major label: ', MAJ_LABEL)
    print('Minor label: ', MIN_LABEL)
    print('Class imbalance: ', counts[MAJ_LABEL]/(counts[MIN_LABEL] if counts[MIN_LABEL] != 0 else 1))
    print('Total labels processed: ', total_counter)
    print('Low noise labels flipped: ', l_flipped_counter)
    print('High noise labels flipped: ', h_flipped_counter)
    print('Lines written to low noise file: ')
    os.system('cat {} | wc -l'.format(filename + '_nar5.csv'))
    print('Lines written to high noise file: ')
    os.system('cat {} | wc -l'.format(filename + '_nar10.csv'))
