#Author: Gentry Atkinson
#Organization: Texas University
#Data: 12 May, 2021
#Generate two Noise Not at Random label sets, 5% and 10%

import numpy as np
from random import randint
import os
from sklearn.neighbors import NearestNeighbors

if __name__ == "__main__":
    clean_labels = np.genfromtxt('data/clean_labels.csv')
    low_noise_labels = np.genfromtxt('data/clean_labels.csv')
    high_noise_labels = np.genfromtxt('data/clean_labels.csv')
    attributes = np.genfromtxt('data/all_attributes.csv', delimiter=',')

    low_noise_file = open('data/nnar_labels_5percent.csv', 'w+')
    high_noise_file = open('data/nnar_labels_10percent.csv', 'w+')

    total_counter = 0
    l_flipped_counter = 0
    h_flipped_counter = 0

    counts = [np.count_nonzero(clean_labels==0), np.count_nonzero(clean_labels==1)]
    MAJ_LABEL = np.argmax(counts)
    MIN_LABEL = np.argmin(counts)
    SET_LENGTH = len(clean_labels)
    print('Majority label: ', MAJ_LABEL)
    print('Minority label: ', MIN_LABEL)

    acc = [attributes[i] for i in range(0, len(attributes), 3)]
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(acc)
    d, i = nbrs.kneighbors(acc)
    print('Number of entries in neighbor table: ', len(i))
    print('Size of neighbor vector: ', len(i[0]))

    while l_flipped_counter < 0.05*SET_LENGTH:
        rand_instance_index = randint(0, SET_LENGTH-1)
        if low_noise_labels[rand_instance_index] == MAJ_LABEL:
            if randint(0,100) < 3+(30 if low_noise_labels[i[rand_instance_index][1]]==MIN_LABEL else 0):
                #print('Mislabel rate: ', 3+(30 if low_noise_labels[i[rand_instance_index][1]]==MIN_LABEL else 0))
                low_noise_labels[rand_instance_index] = MIN_LABEL
                l_flipped_counter += 1

    while h_flipped_counter < 0.1*SET_LENGTH:
        rand_instance_index = randint(0, SET_LENGTH-1)
        if high_noise_labels[rand_instance_index] == MAJ_LABEL:
            if randint(0,100) < 5+(50 if high_noise_labels[i[rand_instance_index][1]]==MIN_LABEL else 0):
                #print('Mislabel rate: ', 5+(50 if high_noise_labels[i[rand_instance_index][1]]==MIN_LABEL else 0))
                high_noise_labels[rand_instance_index] = MIN_LABEL
                h_flipped_counter += 1


    low_noise_file.write('\n'.join([str(i) for i in low_noise_labels]))
    high_noise_file.write('\n'.join([str(i) for i in high_noise_labels]))


    low_noise_file.close()
    high_noise_file.close()

    #sanity checks
    print('Total labels processed: ', total_counter)
    print('Low noise labels flipped: ', l_flipped_counter)
    print('High noise labels flipped: ', h_flipped_counter)
    print('Lines written to low noise file: ')
    os.system('cat data/nar_labels_5percent.csv | wc -l')
    print('Lines written to high noise file: ')
    os.system('cat data/nar_labels_10percent.csv | wc -l')
