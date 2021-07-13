#Author: Gentry Atkinson
#Organization: Texas University
#Data: 12 July, 2021
#Read the 6 datasets, write in nice format, and then write noisy label settings
#This is going to be a big one_hot_encode

#6 Datasets: 2 synthetic, 2 HAR, 2 BioSignal
#Each dataset will have 7 label sets:
#   -clean
#   -low noise NCAR
#   -high noise NCAR
#   -low noise NAR
#   -high noise NAR
#   -low noise NNAR
#   -high noise NNAR

from utils.gen_ts_data import generate_pattern_data_as_dataframe
from utils.add_ncar import add_ncar
from utils.add_nar import add_nar
from utils.add_nnar import add_nnar
import numpy as np

PATH = 'src/data/processed_datasets/'

if(__name__ == "__main__"):

    #Create Synthetic Set 1
    print("### Preparing Dataset: SS1 ###")
    ss1_attributes, ss1_labels_clean = generate_pattern_data_as_dataframe(length=150, numSamples=10000, numClasses=2, percentError=0)
    ss1_attributes = np.reshape(np.array(ss1_attributes['x']),(10000, 150))
    np.savetxt(PATH + 'ss1_attributes.csv', ss1_attributes,  delimiter=',')
    np.savetxt(PATH + 'ss1_labels_clean.csv', ss1_labels_clean, delimiter=',', fmt='%d')

    #Create label sets for SS1
    add_ncar(ss1_labels_clean, PATH + 'ss1_labels', 2)
    add_nar(ss1_labels_clean, PATH + 'ss1_labels', 2)
    add_nnar(ss1_attributes, ss1_labels_clean, PATH + 'ss1_labels', 2)

    #Create Synthetic Set 2
    #Create label sets for SS2

    #Use Lee's files to get HAR Set 1
    #Create label sets for HAR1

    #Process UCI HAR inertial signals into a good file
    #Create label sets for HAR2

    #Process Sleep Apnea set into BioSignal Set 1
    #Create label sets for BS1

    #Process PD Gait set into BioSignal Set 2
    #Create label sets for BS2
