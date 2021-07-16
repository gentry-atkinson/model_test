#Author: Gentry Atkinson
#Organization: Texas University
#Data: 12 May, 2021
#Generate two Noise Not at Random label sets, 5% and 10%

import numpy as np
from random import randint
import os
from sklearn.neighbors import NearestNeighbors

def add_nnar(attributes, clean_labels, filename, num_classes, num_channels=1, att_file=""):
    low_noise_labels = np.copy(clean_labels)
    high_noise_labels = np.copy(clean_labels)

    if attributes == []:
        print("reading attribute file")
        attributes = np.genfromtxt(att_file, delimiter=',')

    low_indexes = open(filename + '_nnar5_indexes.csv', 'w+')
    high_indexes = open(filename + '_nnar10_indexes.csv', 'w+')

    low_noise_file = open(filename + '_nnar5.csv', 'w+')
    high_noise_file = open(filename + '_nnar10.csv', 'w+')

    total_counter = 0
    l_flipped_counter = 0
    h_flipped_counter = 0

    counts = [np.count_nonzero(clean_labels==i) for i in range(num_classes)]
    MAJ_LABEL = np.argmax(counts)
    MIN_LABEL = np.argmin(counts)
    SET_LENGTH = len(clean_labels)
    print('Majority label: ', MAJ_LABEL)
    print('Minority label: ', MIN_LABEL)
    print('Number of labels: ', SET_LENGTH)

    nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(attributes)
    d, i = nbrs.kneighbors(attributes)
    print('Number of entries in neighbor table: ', len(i))
    print('Size of neighbor vector: ', len(i[0]))

    #TODO figure out multi channel data!!!
    while l_flipped_counter < 0.05*SET_LENGTH:
        rand_instance_index = randint(0, SET_LENGTH-1)
        total_counter += 1
        if low_noise_labels[rand_instance_index] == MAJ_LABEL:
            if randint(0,100) < 3+(30 if low_noise_labels[i[rand_instance_index][1]//num_channels]!=MAJ_LABEL else 0):
                #print('Mislabel rate: ', 3+(30 if low_noise_labels[i[rand_instance_index][1]]==MIN_LABEL else 0))
                low_noise_labels[rand_instance_index] = MIN_LABEL
                l_flipped_counter += 1
                low_indexes.write('{}\n'.format(rand_instance_index))
                #print("Low noise flips: ", l_flipped_counter)

    while h_flipped_counter < 0.1*SET_LENGTH:
        rand_instance_index = randint(0, SET_LENGTH-1)
        total_counter += 1
        if high_noise_labels[rand_instance_index] == MAJ_LABEL:
            if randint(0,100) < 5+(50 if high_noise_labels[i[rand_instance_index][1]//num_channels]!=MAJ_LABEL else 0):
                #print('Mislabel rate: ', 5+(50 if high_noise_labels[i[rand_instance_index][1]]==MIN_LABEL else 0))
                high_noise_labels[rand_instance_index] = MIN_LABEL
                h_flipped_counter += 1
                high_indexes.write('{}\n'.format(rand_instance_index))
                #print("High noise flips: ", h_flipped_counter)


    low_noise_file.write('\n'.join([str(i) for i in low_noise_labels]))
    high_noise_file.write('\n'.join([str(i) for i in high_noise_labels]))
    low_noise_file.write('\n')
    high_noise_file.write('\n')


    low_noise_file.close()
    high_noise_file.close()

    #sanity checks
    print('---NNAR---')
    print('Total labels processed: ', total_counter)
    print('Low noise labels flipped: ', l_flipped_counter)
    print('High noise labels flipped: ', h_flipped_counter)
    print('Length of low noise label set', len(low_noise_labels))
    print('Length of high noise label set', len(high_noise_labels))
    print('Lines written to low noise file: ')
    os.system('cat {} | wc -l'.format(filename + '_nnar5.csv'))
    print('Lines written to high noise file: ')
    os.system('cat {} | wc -l'.format(filename + '_nnar10.csv'))
