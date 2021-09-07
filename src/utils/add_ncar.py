#Author: Gentry Atkinson
#Organization: Texas University
#Data: 11 May, 2021
#Generate two Noise Completely at Random label sets, 5% and 10%

import numpy as np
from random import randint
import os

def new_label(old_label, num_classes):
    n = old_label
    while(n==old_label):
        n = randint(0, num_classes-1)
    return n

"""
Noise Completely at Random-> the mislabeling rate is independent of class and features

Apply a flat mislabeling rate to all instances in a dataset. The low noise label set
will have a uniform 5% mislabeling rate. The high noise label set will have a uniform
10% mislabel rate.
"""

def add_ncar(clean_labels, filename, num_classes):
    low_noise_labels = open(filename + '_ncar5.csv', 'w+')
    high_noise_labels = open(filename + '_ncar10.csv', 'w+')
    low_indexes = open(filename + '_ncar5_indexes.csv', 'w+')
    high_indexes = open(filename + '_ncar10_indexes.csv', 'w+')

    total_counter = 0
    l_flipped_counter = 0
    h_flipped_counter = 0

    for i,l in enumerate(clean_labels):
        total_counter += 1
        if randint(0,100)<5:
            low_noise_labels.write('{}\n'.format(new_label(l, num_classes)))
            low_indexes.write('{}\n'.format(int(i)))
            l_flipped_counter += 1
        else:
            low_noise_labels.write('{}\n'.format(int(l)))

        if randint(0,100)<10:
            high_noise_labels.write('{}\n'.format(new_label(l, num_classes)))
            high_indexes.write('{}\n'.format(int(i)))
            h_flipped_counter += 1
        else:
            high_noise_labels.write('{}\n'.format(int(l)))

    low_noise_labels.close()
    high_noise_labels.close()

    #sanity checks
    print('---NCAR---')
    print('Total labels on file: ', len(clean_labels))
    print('Total labels processed: ', total_counter)
    print('Low noise labels flipped: ', l_flipped_counter)
    print('High noise labels flipped: ', h_flipped_counter)
    print('Lines written to low noise file: ')
    os.system('cat {} | wc -l'.format(filename + '_ncar5.csv'))
    print('Lines written to high noise file: ')
    os.system('cat {} | wc -l'.format(filename + '_ncar10.csv'))
