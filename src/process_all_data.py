#Author: Gentry Atkinson
#Organization: Texas State University
#Data: 12 July, 2021
#Read the 6 datasets, write in nice format, and then write noisy label settings
#This is going to be a big one (Narrator: it was)

"""
6 Datasets: 2 synthetic, 2 HAR, 2 BioSignal
Each dataset will have 7 label sets:
  -clean
  -low noise NCAR
  -high noise NCAR
  -low noise NAR
  -high noise NAR
  -low noise NNAR
  -high noise NNAR
"""

from utils.gen_ts_data import generate_pattern_data_as_dataframe
from utils.add_ncar import add_ncar
from utils.add_nar import add_nar
from utils.add_nnar import add_nnar
from data.e4_wristband_Nov2019.e4_load_dataset import e4_load_dataset
from utils.import_datasets import get_uci_data, get_uci_test
import numpy as np
from scipy.signal import resample
from sklearn.utils import shuffle
import os
#import wfdb

PATH = 'src/data/processed_datasets/'

#Use these bools to turn processing of sections on or off
RUN_SS = False
RUN_HAR = False
#RUN_BS = False
RUN_SN = True

if(__name__ == "__main__"):
    if not os.path.isdir(PATH):
            os.system('mkdir src/data/processed_datasets')

    if RUN_SS:
        #Create Synthetic Set 1
        """
        Synthetic Set 1
        Generated using the gen_ts_data script in the utils directory.
        2 classes
        1 channel
        150 samples in every instance
        8000 train instances
        2000 test instances
        """
        print("##### Preparing Dataset: SS1 #####")
        attributes, labels_clean = generate_pattern_data_as_dataframe(length=150, numSamples=10000, numClasses=2, percentError=0)
        attributes = np.reshape(np.array(attributes['x']),(10000, 150))
        attributes, labels_clean = shuffle(attributes, labels_clean, random_state=1899)
        np.savetxt(PATH + 'ss1_attributes_train.csv', attributes[0:8000],  delimiter=',')
        np.savetxt(PATH + 'ss1_labels_clean.csv', labels_clean[0:8000], delimiter=',', fmt='%d')
        np.savetxt(PATH + 'ss1_attributes_test.csv', attributes[8000:10000],  delimiter=',')
        np.savetxt(PATH + 'ss1_labels_test_clean.csv', labels_clean[8000:10000], delimiter=',', fmt='%d')

        #Create label sets for SS1
        add_ncar(labels_clean[0:8000], PATH + 'ss1_labels', 2)
        add_nar(labels_clean[0:8000], PATH + 'ss1_labels', 2)
        add_nnar(attributes[0:8000], labels_clean[0:8000], PATH + 'ss1_labels', 2)

        add_ncar(labels_clean[8000:10000], PATH + 'ss1_labels_test', 2)
        add_nar(labels_clean[8000:10000], PATH + 'ss1_labels_test', 2)
        add_nnar(attributes[8000:10000], labels_clean[8000:10000], PATH + 'ss1_labels_test', 2)

        #Create Synthetic Set 2
        print("##### Preparing Dataset: SS2 #####")
        """
        Synthetic Set 2
        Generated using the gen_ts_data script in the utils directory.
        5 classes
        1 channel
        150 samples in every instance
        24000 train instances
        6000 test instances
        """
        attributes, labels_clean = generate_pattern_data_as_dataframe(length=150, numSamples=30000, numClasses=5, percentError=0)
        attributes = np.reshape(np.array(attributes['x']),(30000, 150))
        attributes, labels_clean = shuffle(attributes, labels_clean, random_state=1899)
        np.savetxt(PATH + 'ss2_attributes_train.csv', attributes[0:24000],  delimiter=',')
        np.savetxt(PATH + 'ss2_labels_clean.csv', labels_clean[0:24000], delimiter=',', fmt='%d')
        np.savetxt(PATH + 'ss2_attributes_test.csv', attributes[24000:30000],  delimiter=',')
        np.savetxt(PATH + 'ss2_labels_test_clean.csv', labels_clean[24000:30000], delimiter=',', fmt='%d')

        #Create label sets for SS2
        add_ncar(labels_clean[0:24000], PATH + 'ss2_labels', 5)
        add_nar(labels_clean[0:24000], PATH + 'ss2_labels', 5)
        add_nnar(attributes[0:24000], labels_clean[0:24000], PATH + 'ss2_labels', 5)

        add_ncar(labels_clean[24000:30000], PATH + 'ss2_labels_test', 5)
        add_nar(labels_clean[24000:30000], PATH + 'ss2_labels_test', 5)
        add_nnar(attributes[24000:30000], labels_clean[24000:30000], PATH + 'ss2_labels_test', 5)
        print("Done with SS")

    if RUN_HAR:
        print("##### Preparing Dataset: HAR1 #####")
        """
        Human Activity Recognition Set 1
        Collected using an E4 data collection device at Texas State University.
        6 classes
        1 channel (total acceleration)
        150 samples in every instance
        2077 train instances
        1091 test instances
        """
        #Use Lee's files to get HAR Set 1
        #Use one_hot_encode to get numerical labels
        attributes, labels_clean, att_test, lab_test = map(np.array, e4_load_dataset(verbose=False, one_hot_encode = True))
        # attributes = np.array(attributes)
        # att_test = np.array(att_test)
        # label_dic = {
        #     'Downstairs':0,
        #     'Jogging':1,
        #     'Not_Labeled':2,
        #     'Sitting':3,
        #     'Standing':4,
        #     'Upstairs':5,
        #     'Walking':6,
        # }
        # labels_clean = np.array([label_dic[i[0]] for i in labels_clean])
        # lab_test = np.array([label_dic[i[0]] for i in lab_test])
        labels_clean = np.argmax(labels_clean, axis=-1)
        lab_test = np.argmax(lab_test, axis=-1)
        num_instances = attributes.shape[0]
        num_samples = attributes.shape[1]
        num_channels = attributes.shape[2]
        print('Shape of e4 data: ', attributes.shape)
        print('Shape of e4 labels: ', labels_clean.shape)
        print('Number of e4 instances: ', len(attributes))
        print('Number of e4 labels: ', len(labels_clean))
        print('Shape of  e4 attributes: ', np.array(attributes).shape)

        attributes = np.reshape(attributes,(num_instances*num_channels, num_samples))
        num_instances = att_test.shape[0]
        num_samples = att_test.shape[1]
        num_channels = att_test.shape[2]
        print('Number of test instances: ', len(att_test))
        print('Number of test labels: ', len(lab_test))
        print('Shape of test attributes: ', np.array(attributes).shape)
        att_test = np.reshape(att_test,(num_instances*num_channels, num_samples))
        np.savetxt(PATH + 'har1_attributes_train.csv', attributes,  delimiter=',')
        np.savetxt(PATH + 'har1_labels_clean.csv', labels_clean, delimiter=',', fmt='%d')
        np.savetxt(PATH + 'har1_attributes_test.csv', att_test,  delimiter=',')
        np.savetxt(PATH + 'har1_labels_test_clean.csv', lab_test, delimiter=',', fmt='%d')

        #Create label sets for HAR1
        add_ncar(labels_clean, PATH + 'har1_labels', 6)
        add_nar(labels_clean, PATH + 'har1_labels', 6)
        add_nnar(attributes, labels_clean, PATH + 'har1_labels', 6, num_channels=1)

        add_ncar(lab_test, PATH + 'har1_labels_test', 6)
        add_nar(lab_test, PATH + 'har1_labels_test', 6)
        add_nnar(att_test, lab_test, PATH + 'har1_labels_test', 6, num_channels=1)

        print("##### Preparing Dataset: HAR2 #####")
        """
        Human Activity Recognition Set 2
        UCI HAR: collected at UC Irving using smartphone
        6 classes
        3 channel (xyz acceleration)
        128 samples in every instance
        7352 train instances
        2947 test instances
        """
        #Process UCI HAR inertial signals into a good file
        attributes, labels_clean, labels = get_uci_data()
        #attributes, labels_clean = shuffle(attributes, labels_clean, random_state=1899)
        print("Shape of UCI data: ", attributes.shape)
        print("Shape of UCI labels: ", labels_clean.shape)
        attributes = np.reshape(np.array(attributes), (7352*3, 128))
        np.savetxt(PATH + 'har2_attributes_train.csv', attributes,  delimiter=',')
        np.savetxt(PATH + 'har2_labels_clean.csv', labels_clean, delimiter=',', fmt='%d')

        #Create label sets for HAR2
        add_ncar(labels_clean, PATH + 'har2_labels', 6)
        add_nar(labels_clean, PATH + 'har2_labels', 6)
        add_nnar(attributes, labels_clean, PATH + 'har2_labels', 6, num_channels=3)

        #Create test sets for HAR2
        attributes, labels_clean, labels = get_uci_test()
        print("Shape of UCI test: ", attributes.shape)
        print("Shape of UCI test: ", labels_clean.shape)
        attributes = np.reshape(attributes,(2947*3, 128))
        np.savetxt(PATH + 'har2_attributes_test.csv', attributes,  delimiter=',')
        np.savetxt(PATH + 'har2_labels_test_clean.csv', labels_clean, delimiter=',', fmt='%d')

        add_ncar(labels_clean, PATH + 'har2_labels_test', 6)
        add_nar(labels_clean, PATH + 'har2_labels_test', 6)
        add_nnar(attributes, labels_clean, PATH + 'har2_labels_test', 6, num_channels=3)
        print("Done with HAR")

    if RUN_SN:
        #Create Synthetic Set 1
        """
        Sensor Network Set 1
        Rainfall in Australia dataset from:
            http://www.bom.gov.au/climate/change/datasets/datasets.shtml
        2 classes
        # channel
        # samples in every instance
        # train instances
        # test instances
        """
        print("##### Preparing Dataset: SN1 #####")
        weather_file = 'src/data/rain_in_australia/weatherAUS.csv'
        locations = []
        with open(weather_file) as input:
            print(input.readline().split(','))
            for l in input:
                l = l.split(',')
                if len(l)>1:
                    if l[1] not in locations:
                        locations.append(l[1])
        print('\n'.join(sorted(locations)))
        print('Number of locations: ', len(locations))
