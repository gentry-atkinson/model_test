#Author: Gentry Atkinson
#Organization: Texas University
#Data: 11 May, 2021
#Generate two Noise Completely at Random label sets, 5% and 10%

import numpy as np
from random import randint
import os

if __name__ == "__main__":
    clean_labels = np.genfromtxt('data/clean_labels.csv')
    low_noise_labels = open('data/ncar_labels_5percent.csv', 'w+')
    high_noise_labels = open('data/ncar_labels_10percent.csv', 'w+')

    total_counter = 0
    l_flipped_counter = 0
    h_flipped_counter = 0

    for l in clean_labels:
        total_counter += 1
        if randint(0,100)<5:
            low_noise_labels.write('{}\n'.format(1 if l==0 else 0))
            l_flipped_counter += 1
        else:
            low_noise_labels.write('{}\n'.format(l))

        if randint(0,100)<10:
            high_noise_labels.write('{}\n'.format(1 if l==0 else 0))
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
    os.system('cat data/ncar_labels_5percent.csv | wc -l')
    print('Lines written to high noise file: ')
    os.system('cat data/ncar_labels_10percent.csv | wc -l')
