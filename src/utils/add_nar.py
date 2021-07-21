#Author: Gentry Atkinson
#Organization: Texas University
#Data: 12 May, 2021
#Generate two Noise at Random label sets, 5% and 10%

import numpy as np
from random import randint
import os

def add_nar(clean_labels, filename, num_classes):
    low_noise_labels = open(filename + '_nar5.csv', 'w+')
    high_noise_labels = open(filename + '_nar10.csv', 'w+')
    low_indexes = open(filename + '_nar5_indexes.csv', 'w+')
    high_indexes = open(filename + '_nar10_indexes.csv', 'w+')

    total_counter = 0
    l_flipped_counter = 0
    h_flipped_counter = 0

    counts = [np.count_nonzero(clean_labels==i) for i in range(num_classes)]
    MAJ_LABEL = np.argmax(counts)
    MIN_LABEL = np.argmin(counts)

    imbalance = len(clean_labels)/counts[MAJ_LABEL]


    for i,l in enumerate(clean_labels):
        total_counter += 1
        if l==MAJ_LABEL and randint(0,100)<5*imbalance:
            low_noise_labels.write('{}\n'.format(MIN_LABEL))
            low_indexes.write('{}\n'.format(i))
            l_flipped_counter += 1
        else:
            low_noise_labels.write('{}\n'.format(l))

        if l==MAJ_LABEL and randint(0,100)<10*imbalance:
            high_noise_labels.write('{}\n'.format(MIN_LABEL))
            high_indexes.write('{}\n'.format(i))
            h_flipped_counter += 1
        else:
            high_noise_labels.write('{}\n'.format(l))


    low_noise_labels.close()
    high_noise_labels.close()

    #sanity checks
    print('---NAR---')
    print('Major label: ', MAJ_LABEL)
    print('Minor label: ', MIN_LABEL)
    print('Class imbalance: ', counts[MAJ_LABEL]/(counts[MIN_LABEL] if counts[MIN_LABEL] != 0 else 1)
    print('Total labels processed: ', total_counter)
    print('Low noise labels flipped: ', l_flipped_counter)
    print('High noise labels flipped: ', h_flipped_counter)
    print('Lines written to low noise file: ')
    os.system('cat {} | wc -l'.format(filename + '_nar5.csv'))
    print('Lines written to high noise file: ')
    os.system('cat {} | wc -l'.format(filename + '_nar10.csv'))
