#Author: Gentry Atkinson
#Organization: Texas State University
#Data: 12 July, 2021
#Updating: 11 Nov, 2022
#Read the 6 datasets, write in nice format, and then write noisy label settings
#This is going to be a big one (Narrator: it was)

#TODO:
#  -add a biosignal dataset

"""
6 Datasets: 2 synthetic, 2 HAR, 2 BioSignal
Each dataset will have 31 label sets:
  - clean
  - 1% to 20% NCAR
  - 1% to 20% NAR
  - 1% to 20% NNAR
"""

MIN_PERCENT = 1
MAX_PERCENT = 15

import os

#from utils.import_datasets import get_uci_data, get_uci_test
import numpy as np
import pandas as pd
import tensorflow as tf
#from cmath import isinf
from attr import attr
from sklearn.preprocessing import minmax_scale
#from scipy.signal import resample
from sklearn.utils import shuffle

from data.load_data_time_series.HAR.e4_wristband_Nov2019.e4_load_dataset import \
    e4_load_dataset
from data.load_data_time_series.HAR.UCI_HAR.uci_har_load_dataset import \
    uci_har_load_dataset
from utils.add_nar import add_nar
from utils.add_ncar import add_ncar
from utils.add_nnar import add_nnar
from utils.gen_ts_data import generate_pattern_data_as_array
from utils.ts_feature_toolkit import clean_nan_and_inf

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))   


PATH = 'src/data/processed_datasets/'

#Use these bools to turn processing of sections on or off
RUN_SS = True
RUN_HAR = False
#RUN_BS = False
RUN_SN = False

def record_sn_instance(att_array, table, feature_list, start_index, end_index):
    for f in feature_list:
        att_array.append((table[f][start_index:end_index]))

def load_synthetic_dataset(
        num_train : int,
        num_test : int,
        num_classes : int,
        instance_len : int
    ):

    params = {
        'avg_pattern_length' : [],
        'avg_amplitude' : [],
        'default_variance' : [],
        'variance_pattern_length' : [],
        'variance_amplitude' : []
    }

    for _ in range(num_classes):
        params['avg_amplitude'].append(np.random.randint(0, 5))
        params['avg_pattern_length'].append(np.random.randint(5, 15))
        params['default_variance'].append(np.random.randint(1, 4))
        params['variance_pattern_length'].append(np.random.randint(5, 20))
        params['variance_amplitude'].append(np.random.randint(1, 5))

    train_set = np.zeros((num_train, instance_len))
    test_set = np.zeros((num_test, instance_len))

    train_labels = []
    test_labels = []

    train_label_count = [0]*num_classes
    test_label_count = [0]*num_classes

    for i in range (num_train):
        label = np.random.randint(0, num_classes)
        train_labels.append(label)
        train_set[i, :] = generate_pattern_data_as_array(
            length=instance_len,
            avg_pattern_length=params['avg_pattern_length'][label],
            avg_amplitude=params['avg_amplitude'][label],
            default_variance=params['default_variance'][label],
            variance_pattern_length=params['variance_pattern_length'][label],
            variance_amplitude=params['variance_amplitude'][label]
        )
        train_label_count[label] += 1

    for i in range (num_test):
        label = np.random.randint(0, num_classes)
        test_labels.append(label)
        test_set[i, :] = generate_pattern_data_as_array(
            length=instance_len,
            avg_pattern_length=params['avg_pattern_length'][label],
            avg_amplitude=params['avg_amplitude'][label],
            default_variance=params['default_variance'][label],
            variance_pattern_length=params['variance_pattern_length'][label],
            variance_amplitude=params['variance_amplitude'][label]
        )
        test_label_count[label] += 1


    train_set = np.reshape(train_set, (train_set.shape[0], train_set.shape[1], 1))
    test_set = np.reshape(test_set, (test_set.shape[0], test_set.shape[1], 1))

    train_labels = np.array(train_labels, dtype='int')
    test_labels = np.array(test_labels, dtype='int')

    print("Train labels: ", '\n'.join([str(i) for i in train_label_count]))
    print("Test labels: ", '\n'.join([str(i) for i in test_label_count]))

    print("Train data shape: ", train_set.shape)
    print("Test data shape: ", test_set.shape)

    return train_set, train_labels, test_set, test_labels

def run_ss():
    #Create Synthetic Set 1
    """
    Synthetic Set 1
    Generated using the gen_ts_data script in the utils directory.
    2 classes
    1 channel
    150 samples in every instance
    8001 train instances
    2001 test instances
    """
    print("##### Preparing Dataset: SS1 #####")
    X_train, y_train, X_test, y_test = load_synthetic_dataset(8001, 2001, 2, 150)
    np.save(PATH + 'ss1_attributes_train.npy', X_train)
    np.save(PATH + 'ss1_labels_clean.npy', y_train)
    np.save(PATH + 'ss1_attributes_test.npy', X_test)
    np.save(PATH + 'ss1_labels_test_clean.npy', y_test)

    #Create label sets for SS1
    for mislab_rate in range(MIN_PERCENT, MAX_PERCENT+1):
        add_ncar(y_train, PATH + 'ss1_labels', 2, mislab_rate)
        add_nar(y_train, PATH + 'ss1_labels', 2, mislab_rate)
        add_nnar(X_train, y_train, PATH + 'ss1_labels', 2, mislab_rate)

        add_ncar(y_test, PATH + 'ss1_labels_test', 2, mislab_rate)
        add_nar(y_test, PATH + 'ss1_labels_test', 2, mislab_rate)
        add_nnar(X_test, y_test, PATH + 'ss1_labels_test', 2, mislab_rate)

    #Create Synthetic Set 2
    print("##### Preparing Dataset: SS2 #####")
    """
    Synthetic Set 2
    Generated using the gen_ts_data script in the utils directory.
    3 classes
    1 channel
    150 samples in every instance
    24000 train instances
    6000 test instances
    """
    # attributes, labels_clean = generate_pattern_data_as_dataframe(length=150, numSamples=30000, numClasses=5, percentError=0)
    # attributes = np.reshape(np.array(attributes['x']),(30000, 150))
    # attributes, labels_clean = shuffle(attributes, labels_clean, random_state=1899)
    X_train, y_train, X_test, y_test = load_synthetic_dataset(24000, 6000, 3, 150)
    #np.savetxt(PATH + 'ss2_attributes_train.csv', attributes[0:24000],  delimiter=',')
    np.save(PATH + 'ss2_attributes_train.npy', X_train)
    #np.savetxt(PATH + 'ss2_labels_clean.csv', labels_clean[0:24000], delimiter=',', fmt='%d')
    np.save(PATH + 'ss2_labels_clean.npy', y_train)
    #np.savetxt(PATH + 'ss2_attributes_test.csv', attributes[24000:30000],  delimiter=',')
    np.save(PATH + 'ss2_attributes_test.npy', X_test)
    #np.savetxt(PATH + 'ss2_labels_test_clean.csv', labels_clean[24000:30000], delimiter=',', fmt='%d')
    np.save(PATH + 'ss2_labels_test_clean.npy', y_test)

    #Create label sets for SS2
    for mislab_rate in range(MIN_PERCENT, MAX_PERCENT+1):
        add_ncar(y_train, PATH + 'ss2_labels', 3, mislab_rate)
        add_nar(y_train, PATH + 'ss2_labels', 3, mislab_rate)
        add_nnar(X_train, y_train, PATH + 'ss2_labels', 3, mislab_rate)

        add_ncar(y_test, PATH + 'ss2_labels_test', 3, mislab_rate)
        add_nar(y_test, PATH + 'ss2_labels_test', 3, mislab_rate)
        add_nnar(X_test, y_test, PATH + 'ss2_labels_test', 3, mislab_rate)
    print("Done with SS")

def run_har():
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
    X_train, y_train, X_test, y_test = e4_load_dataset(
        verbose=False, one_hot_encode = True, incl_xyz_accel=False, 
        incl_rms_accel=True, incl_val_group=False
    )

    y_train = np.argmax(y_train, axis=-1)
    y_test = np.argmax(y_test, axis=-1)
    num_instances = X_train.shape[0]
    num_samples = X_train.shape[1]
    num_channels = X_train.shape[2]
    print('Shape of e4 train data: ', X_train.shape)
    print('Shape of e4 train labels: ', y_train.shape)
    print('Shape of e4 test data: ', X_test.shape)
    print('Shape of e4 test labels: ', y_test.shape)

    #np.savetxt(PATH + 'har1_attributes_train.csv', attributes,  delimiter=',')
    np.save(PATH + 'har1_attributes_train.npy', X_train)
    #np.savetxt(PATH + 'har1_labels_clean.csv', labels_clean, delimiter=',', fmt='%d')
    np.save(PATH + 'har1_labels_clean.npy', y_train)
    #np.savetxt(PATH + 'har1_attributes_test.csv', att_test,  delimiter=',')
    np.save(PATH + 'har1_attributes_test.npy', X_test)
    #np.savetxt(PATH + 'har1_labels_test_clean.csv', lab_test, delimiter=',', fmt='%d')
    np.save(PATH + 'har1_labels_test_clean.npy', y_test)

    #Create label sets for HAR1
    for mislab_rate in range(MIN_PERCENT, MAX_PERCENT+1):
        add_ncar(y_train, PATH + 'har1_labels', 6, mislab_rate)
        add_nar(y_train, PATH + 'har1_labels', 6, mislab_rate)
        add_nnar(X_train, y_train, PATH + 'har1_labels', 6, num_channels=1, mislab_rate=mislab_rate)

        add_ncar(y_test, PATH + 'har1_labels_test', 6, mislab_rate)
        add_nar(y_test, PATH + 'har1_labels_test', 6, mislab_rate)
        add_nnar(X_test, y_test, PATH + 'har1_labels_test', 6, num_channels=1, mislab_rate=mislab_rate)

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
    #attributes, labels_clean, labels = get_uci_data()
    #attributes, labels_clean = shuffle(attributes, labels_clean, random_state=1899)
    X_train, y_train, X_test, y_test = uci_har_load_dataset(
        verbose=False, one_hot_encode = True, incl_xyz_accel=True, 
        incl_rms_accel=False, incl_val_group=False
    )

    y_train = np.argmax(y_train, axis=-1)
    y_test = np.argmax(y_test, axis=-1)

    print("Shape of UCI data: ", X_train.shape)
    print("Shape of UCI labels: ", y_train.shape)
    print("Shape of UCI test: ", X_test.shape)
    print("Shape of UCI test: ", y_test.shape)
    #attributes = np.reshape(np.array(attributes), (7352*3, 128))
    #np.savetxt(PATH + 'har2_attributes_train.csv', attributes,  delimiter=',')
    np.save(PATH + 'har2_attributes_train.npy', X_train)
    #np.savetxt(PATH + 'har2_labels_clean.csv', labels_clean, delimiter=',', fmt='%d')
    np.save(PATH + 'har2_labels_clean.npy', y_train)

    

    # #Create test sets for HAR2
    # attributes, labels_clean, labels = get_uci_test()
    # print("Shape of UCI test: ", attributes.shape)
    # print("Shape of UCI test: ", labels_clean.shape)
    # attributes = np.reshape(attributes,(2947*3, 128))
    # np.savetxt(PATH + 'har2_attributes_test.csv', attributes,  delimiter=',')
    np.save(PATH + 'har2_attributes_test.npy', X_test)
    # np.savetxt(PATH + 'har2_labels_test_clean.csv', labels_clean, delimiter=',', fmt='%d')
    np.save(PATH + 'har2_labels_test_clean.npy', y_test)

    #Create label sets for HAR2
    for mislab_rate in range(MIN_PERCENT, MAX_PERCENT+1):
        add_ncar(y_train, PATH + 'har2_labels', 6, mislab_rate)
        add_nar(y_train, PATH + 'har2_labels', 6, mislab_rate)
        add_nnar(X_train, y_train, PATH + 'har2_labels', 6, num_channels=3, mislab_rate=mislab_rate)

        add_ncar(y_test, PATH + 'har2_labels_test', 6, mislab_rate)
        add_nar(y_test, PATH + 'har2_labels_test', 6, mislab_rate)
        add_nnar(X_test, y_test, PATH + 'har2_labels_test', 6, num_channels=3, mislab_rate=mislab_rate)
    print("Done with HAR")

def run_sn():
    #Create Sensor Network 1
    """
    Sensor Network Set 1
    Rainfall in Australia dataset from:
        http://www.bom.gov.au/climate/change/datasets/datasets.shtml
    2 classes
    6 channel
    30 samples in every instance
    113899 train instances
    30091 test instances
    """
    INSTANCE_LEN = 30
    TRAIN_SPLIT = 0.8

    print("##### Preparing Dataset: SN1 #####")
    weather_file = 'src/data/rain_in_australia/weatherAUS.csv'

    weather_table = pd.read_csv(weather_file)
    locations = set(weather_table['Location'])
    num_train_locs = int(TRAIN_SPLIT*len(locations))
    print('Number of locations: ', len(locations))
    print('Train Locations:', list(locations)[0:num_train_locs])
    print('Test Locations:', list(locations)[num_train_locs:])

    feature_list = ['MinTemp', 'MaxTemp', 'WindGustDir', 'WindGustSpeed', 'Pressure9am', 'Pressure3pm']

    X_train = []
    X_test = []
    y_train = []
    y_test = []

    #Prepare an ordinal value for wind direction
    wind_dirs = set(weather_table['WindGustDir'])
    dir_dic = {}
    for i,d in enumerate(list(wind_dirs)):
        dir_dic[d] = i
    print('Wind direction dictionary: ', dir_dic)

    train_count = 0
    test_count = 0
    i = 0
    while i < len(weather_table['Location'])-30:
        if weather_table.loc[i]['Location'] == weather_table.loc[i+30]['Location']:
            #Record this instance
            
            if weather_table.loc[i]['Location'] in list(locations)[0:num_train_locs]:
                train_count += 1
                for f in feature_list:
                    line = list()
                    if f == 'WindGustDir':
                        line = np.array(([dir_dic[k] for k in weather_table[f][i:i+30]]))
                    else:
                            line = np.array((weather_table[f][i:i+30]))
                    line = [j if not np.isnan(j) and not np.isinf(j) else 0 for j in line ]
                    # max_val = np.max(line)
                    # line = np.divide(line, max_val if max_val != 0 else 1)
                    X_train.append(line)
                y_train.append(1 if weather_table['RainTomorrow'][i+30]=='Yes' else 0)
            else:
                test_count += 1
                for f in feature_list:
                    line = []
                    if f == 'WindGustDir':
                        line = np.array(([dir_dic[k] for k in weather_table[f][i:i+30]]))
                    else:
                            line = np.array((weather_table[f][i:i+30]))
                    line = [j if not np.isnan(j) and not np.isinf(j) else 0 for j in line ]
                    # max_val = abs(np.max(line))
                    # line = np.divide(line, max_val if max_val != 0 else 1)
                    X_test.append(line)
                y_test.append(1 if weather_table['RainTomorrow'][i+30]=='Yes' else 0)
            i+=1
        else:
            #Skip to next location
            j = i+1
            #print('Next Location ', i)
            while weather_table.loc[i]['Location'] == weather_table.loc[j]['Location']:
                j+= 1
            i=j

    

    y_train = np.array(y_train, dtype='int')
    y_test = np.array(y_test, dtype='int')

    old_len_train = len(X_train)
    old_len_test = len(X_test)
    X_train = minmax_scale(X_train, (-1, 1), axis=1)
    X_test = minmax_scale(X_test, (-1, 1), axis=1)


    X_train = np.reshape(np.array(X_train), (old_len_train//6, 6, 30))
    X_test = np.reshape(np.array(X_test), (old_len_test//6, 6, 30))

    # normalize(attributes, axis=1, copy=False, norm='max')
    # normalize(test_att, axis=1, copy=False, norm='max')
    
    

    # attributes = clean_nan_and_inf(attributes)
    # test_att = clean_nan_and_inf(test_att)

    print ("Number of train instances: ", train_count)
    print ("Number of test instances: ", test_count)
    print ("Number of train array: ", len(X_train))
    print ("Number of test array: ", len(X_test))
    print ("Rainy train days: ", sum(y_train))
    print ("Rainy test days: ", sum(y_test))

    print("Shape of train data: ", X_train.shape)
    print('Sahpe of test data: ', X_test.shape)

    del weather_table

    # print('Mintemp: ', ', '.join([str(i) for i in attributes[0]]))
    # print('MaxTemp: ', ', '.join([str(i) for i in attributes[1]]))
    # print('WindGustDir: ', ', '.join([str(i) for i in attributes[2]]))
    # print('WindGustSpeed: ', ', '.join([str(i) for i in attributes[3]]))
    # print('Pressure9am: ', ', '.join([str(i) for i in attributes[4]]))
    # print('Pressure3pm: ', ', '.join([str(i) for i in attributes[5]]))

    #write attributes to file
    #np.savetxt(PATH + 'sn1_attributes_train.csv', np.array(attributes),  delimiter=',')
    np.save(PATH + 'sn1_attributes_train.npy', X_train)
    #np.savetxt(PATH + 'sn1_attributes_test.csv', np.array(test_att),  delimiter=',')
    np.save(PATH + 'sn1_attributes_test.npy', X_test)
    #np.savetxt(PATH + 'sn1_labels_clean.csv', np.array(labels_clean), delimiter=',', fmt='%d')
    np.save(PATH + 'sn1_labels_clean.npy', y_train)
    #np.savetxt(PATH + 'sn1_labels_test_clean.csv', np.array(labels_test), delimiter=',', fmt='%d')
    np.save(PATH + 'sn1_labels_test_clean.npy', y_test)

    for mislab_rate in range(MIN_PERCENT, MAX_PERCENT+1):
        add_ncar(y_train, PATH + 'sn1_labels', 2, mislab_rate)
        add_nar(y_train, PATH + 'sn1_labels', 2, mislab_rate)
        add_nnar(X_train, y_train, PATH + 'sn1_labels', 2, num_channels=6,mislab_rate=mislab_rate)

        add_ncar(y_test, PATH + 'sn1_labels_test', 2, mislab_rate)
        add_nar(y_test, PATH + 'sn1_labels_test', 2, mislab_rate)
        add_nnar(X_test, y_test, PATH + 'sn1_labels_test', 2, num_channels=6, mislab_rate=mislab_rate)

    #Create Synthetic Set 2
    """
    Sensor Network Set 2
    Occupancy Detection Data Set dataset from:
        Accurate occupancy detection of an office room from light, temperature, humidity and CO2 
        measurements using statistical learning models. Luis M. Candanedo, VÃ©ronique Feldheim. 
        Energy and Buildings. Volume 112, 15 January 2016, Pages 28-39.
    5 classes
    5 channel
    30 samples in every instance, one sample per minute
    # train instances
    # test instances
    """

    INSTANCE_LEN = 30

    room_train_file = "src/data/occupancy/datatraining.txt"
    room_test_file = "src/data/occupancy/datatest.txt"
    
    room_train_table = pd.read_csv(room_train_file)
    room_test_table = pd.read_csv(room_test_file)

    features = ['Temperature',  'Humidity',  'Light', 'CO2',  'HumidityRatio']
    key = 'Occupancy'

    X_train = []
    X_test = []
    y_train = []
    y_test = []

    train_count = 0
    test_count = 0

    #Label Set:
    #0 -> Room empty for full sample            "Empty"
    #1 -> Room occupied for full sample         "Occupied"
    #2 -> Room started empty ended occupied     "Entered"
    #3 -> Room started occupied ended empty     "Exited"
    #4 -> Anything else                         "Partial" 

    for i in range(1, len(room_train_table['Temperature'])-INSTANCE_LEN):
        for f in features:
            line = []
            line.append(room_train_table[f][i:i+INSTANCE_LEN])
            X_train.append(line)
        # if sum(room_train_table[key][i:i+INSTANCE_LEN]) > INSTANCE_LEN/2:
        #     labels_clean.append(1)
        # else:
        #     labels_clean.append(0)
        if sum(room_train_table[key][i:i+INSTANCE_LEN]) == 0:
            y_train.append(0)
        elif sum(room_train_table[key][i:i+INSTANCE_LEN]) == INSTANCE_LEN:
            y_train.append(1)
        elif room_train_table[key][i] == 0 and room_train_table[key][i+INSTANCE_LEN] == 1:
            y_train.append(2)
        elif room_train_table[key][i] == 1 and room_train_table[key][i+INSTANCE_LEN] == 0:
            y_train.append(3)
        else:
            y_train.append(4)

    # print('Number of samples in dataset: ', len(room_train_table['Temperature']))
    # print('Length of attribute array: ', len(attributes))
    # print('Legth of instance: ', len(attributes[0]))
    # print(attributes[0])

    for i in range(1, len(room_test_table['Temperature'])-INSTANCE_LEN):
        for f in features:
            line = []
            line.append(room_test_table[f][i:i+INSTANCE_LEN])
            X_test.append(line)
        # if sum(room_test_table[key][i:i+INSTANCE_LEN]) > INSTANCE_LEN/2:
        #     labels_test.append(1)
        # else:
        #     labels_test.append(0)
        if sum(room_test_table[key][i:i+INSTANCE_LEN]) == 0:
            y_test.append(0)
        elif sum(room_test_table[key][i:i+INSTANCE_LEN]) == INSTANCE_LEN:
            y_test.append(1)
        elif room_test_table[key][i] == 0 and room_test_table[key][i+INSTANCE_LEN] == 1:
            y_test.append(2)
        elif room_test_table[key][i] == 1 and room_test_table[key][i+INSTANCE_LEN] == 0:
            y_test.append(3)
        else:
            y_test.append(4)

    X_train = np.reshape(np.array(X_train), (len(X_train)//5, 5, INSTANCE_LEN))
    X_test =  np.reshape(np.array(X_test), (len(X_test)//5, 5, INSTANCE_LEN))

    y_train = np.array(y_train, dtype='int')
    y_test = np.array(y_test, dtype='int')

    print ("Number of train instances: ", train_count)
    print ("Number of test instances: ", test_count)
    print ("Number of train array: ", len(X_train))
    print ("Number of test array: ", len(X_test))
    print ("Rainy occupied test days: ", np.sum(np.where(y_test==1), axis=None))
    print ("Rainy unoccupied test days: ", np.sum(np.where(y_test!=1), axis=None))

    print("Shape of train data: ", X_train.shape)
    print('Sahpe of test data: ', X_test.shape)

    # attributes = clean_nan_and_inf(attributes)
    # test_att = clean_nan_and_inf(test_att)

    #write attributes to file
    # np.savetxt(PATH + 'sn2_attributes_train.csv', np.array(attributes),  delimiter=',')
    np.save(PATH + 'sn2_attributes_train.npy', X_train)
    # np.savetxt(PATH + 'sn2_attributes_test.csv', np.array(test_att),  delimiter=',')
    np.save(PATH + 'sn2_attributes_test.csv', X_test)
    # np.savetxt(PATH + 'sn2_labels_clean.csv', np.array(labels_clean), delimiter=',', fmt='%d')
    np.save(PATH + 'sn2_labels_clean.csv', y_train)
    # np.savetxt(PATH + 'sn2_labels_test_clean.csv', np.array(labels_test), delimiter=',', fmt='%d')
    np.save(PATH + 'sn2_labels_test_clean.csv', y_test)

    for mislab_rate in range(MIN_PERCENT, MAX_PERCENT+1):
        add_ncar(y_train, PATH + 'sn2_labels', 5, mislab_rate)
        add_nar(y_train, PATH + 'sn2_labels', 5, mislab_rate)
        add_nnar(X_train, y_train, PATH + 'sn2_labels', 5, num_channels=5, mislab_rate=mislab_rate)

        add_ncar(y_test, PATH + 'sn2_labels_test', 5, mislab_rate)
        add_nar(y_test, PATH + 'sn2_labels_test', 5, mislab_rate)
        add_nnar(X_test, y_test, PATH + 'sn2_labels_test', 5, num_channels=5, mislab_rate=mislab_rate)

if(__name__ == "__main__"):
    if not os.path.isdir(PATH):
            os.system('mkdir src/data/processed_datasets')
    if not os.path.isdir('content'):
            os.system('mkdir content')
    if not os.path.isdir('content/temp'):
            os.system('mkdir content/temp')

    np.random.seed(1899)

    if RUN_SS: run_ss()

    if RUN_HAR: run_har()
        
    if RUN_SN: run_sn()

    print('All datasets processed')
        