#Author: Gentry Atkinson
#Organization: Texas University
#Data: 12 May, 2021
#Generate two Noise at Random label sets, 5% and 10%

import numpy as np
from random import randint
import os

if __name__ == "__main__":
    clean_labels = np.genfromtxt('data/clean_labels.csv')
    low_noise_labels = open('data/nar_labels_5percent.csv', 'w+')
    high_noise_labels = open('data/nar_labels_10percent.csv', 'w+')

    total_counter = 0
    l_flipped_counter = 0
    h_flipped_counter = 0

    counts = [np.count_nonzero(clean_labels==0), np.count_nonzero(clean_labels==1)]
    MAJ_LABEL = np.argmax(counts)
    MIN_LABEL = np.argmin(counts)
    print('Majority label: ', MAJ_LABEL)
    print('Minority label: ', MIN_LABEL)
    embalance = len(clean_labels)/counts[MAJ_LABEL]
    print('Class embalance: ', embalance)

    for l in clean_labels:
        total_counter += 1
        if l==MAJ_LABEL and randint(0,100)<5*embalance:
            low_noise_labels.write('{}\n'.format(MIN_LABEL))
            l_flipped_counter += 1
        else:
            low_noise_labels.write('{}\n'.format(l))

        if l==MAJ_LABEL and randint(0,100)<10*embalance:
            high_noise_labels.write('{}\n'.format(MIN_LABEL))
            h_flipped_counter += 1
        else:
            high_noise_labels.write('{}\n'.format(l))


    low_noise_labels.close()
    high_noise_labels.close()

    #sanity checks
    print('Total labels processed: ', total_counter)
    print('Low noise labels flipped: ', l_flipped_counter)
    print('High noise labels flipped: ', h_flipped_counter)
    print('Lines written to low noise file: ')
    os.system('cat data/nar_labels_5percent.csv | wc -l')
    print('Lines written to high noise file: ')
    os.system('cat data/nar_labels_10percent.csv | wc -l')
