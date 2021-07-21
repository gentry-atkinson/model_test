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
from data.e4_wristband_Nov2019.e4_get_x_y_sub import get_X_y_sub
from data.e4_wristband_Nov2019.e4_load_dataset import e4_load_dataset
from import_datasets import get_uci_data, get_uci_test
import numpy as np
import os
import wfdb
from scipy.signal import resample

PATH = 'src/data/processed_datasets/'

RUN_SS = True
RUN_HAR = True
RUN_BS = True

if(__name__ == "__main__"):

    if not os.path.isdir(PATH):
            os.system('mkdir src/data/processed_datasets')

    if RUN_SS:
        #Create Synthetic Set 1
        print("##### Preparing Dataset: SS1 #####")
        attributes, labels_clean = generate_pattern_data_as_dataframe(length=150, numSamples=10000, numClasses=2, percentError=0)
        attributes = np.reshape(np.array(attributes['x']),(10000, 150))
        np.savetxt(PATH + 'ss1_attributes_train.csv', attributes[0:8000],  delimiter=',')
        np.savetxt(PATH + 'ss1_labels_clean.csv', labels_clean[0:8000], delimiter=',', fmt='%d')
        np.savetxt(PATH + 'ss1_attributes_test.csv', attributes[8000:10000],  delimiter=',')
        np.savetxt(PATH + 'ss1_labels_test.csv', labels_clean[8000:10000], delimiter=',', fmt='%d')

        #Create label sets for SS1
        add_ncar(labels_clean[0:8000], PATH + 'ss1_labels', 2)
        add_nar(labels_clean[0:8000], PATH + 'ss1_labels', 2)
        add_nnar(attributes[0:8000], labels_clean[0:8000], PATH + 'ss1_labels', 2)

        #Create Synthetic Set 2
        print("##### Preparing Dataset: SS2 #####")
        attributes, labels_clean = generate_pattern_data_as_dataframe(length=150, numSamples=30000, numClasses=5, percentError=0)
        attributes = np.reshape(np.array(attributes['x']),(30000, 150))
        np.savetxt(PATH + 'ss2_attributes_train.csv', attributes[0:24000],  delimiter=',')
        np.savetxt(PATH + 'ss2_labels_clean.csv', labels_clean[0:24000], delimiter=',', fmt='%d')
        np.savetxt(PATH + 'ss2_attributes_test.csv', attributes[24000:30000],  delimiter=',')
        np.savetxt(PATH + 'ss2_labels_test.csv', labels_clean[24000:30000], delimiter=',', fmt='%d')

        #Create label sets for SS2
        add_ncar(labels_clean[0:24000], PATH + 'ss2_labels', 5)
        add_nar(labels_clean[0:24000], PATH + 'ss2_labels', 5)
        add_nnar(attributes[0:24000], labels_clean[0:24000], PATH + 'ss2_labels', 5)

    if RUN_HAR:
        print("##### Preparing Dataset: HAR1 #####")
        #Use Lee's files to get HAR Set 1
        #attributes, labels_clean, sub, xyinfo = get_X_y_sub(working_dir='src/temp', zip_flist=[
        #    '1574621345_A01F11.zip', '1574622389_A01F11.zip',
        #    '1574624998_A01F11.zip', '1574625540_A01F11.zip'], time_steps=150,
        #    step=50)
        attributes, labels_clean, att_test, lab_test = e4_load_dataset()
        attributes = np.array(attributes)
        att_test = np.array(att_test)
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
        np.savetxt(PATH + 'har1_labels_test.csv', lab_test, delimiter=',', fmt='%d')

        #Create label sets for HAR1
        add_ncar(labels_clean, PATH + 'har1_labels', 7)
        add_nar(labels_clean, PATH + 'har1_labels', 7)
        add_nnar(attributes, labels_clean, PATH + 'har1_labels', 7, num_channels=4)

        print("##### Preparing Dataset: HAR2 #####")
        #Process UCI HAR inertial signals into a good file
        attributes, labels_clean, labels = get_uci_data()
        attributes = np.reshape(np.array(attributes), (7352*3, 128))
        np.savetxt(PATH + 'har2_attributes_train.csv', attributes,  delimiter=',')
        np.savetxt(PATH + 'har2_labels_clean.csv', labels_clean, delimiter=',', fmt='%d')

        #Create label sets for HAR2
        add_ncar(labels_clean, PATH + 'har2_labels', 6)
        add_nar(labels_clean, PATH + 'har2_labels', 6)
        add_nnar(attributes, labels_clean, PATH + 'har2_labels', 6, num_channels=3)

        #Create test sets for HAR2
        attributes, labels_clean, labels = get_uci_test()
        attributes = np.reshape(attributes,(2947*3, 128))
        np.savetxt(PATH + 'har2_attributes_test.csv', attributes,  delimiter=',')
        np.savetxt(PATH + 'har2_labels_test.csv', labels_clean, delimiter=',', fmt='%d')

    if RUN_BS:
        print("##### Preparing Dataset: BS1 #####")
        #Process Sleep Apnea set into BioSignal Set 1
        #The instances get downsampled from 6000 to 600 because otherwise it all
        #falls apart
        attributes = []
        labels_clean = []
        os.system('rm {}'.format(PATH + 'bs1_attributes_train.csv'))
        os.system('rm {}'.format(PATH + 'bs1_labels_clean.csv'))
        att_file = open(PATH + 'bs1_attributes_train.csv', 'a+')
        lab_file = open(PATH + 'bs1_labels_clean.csv', 'a+')
        with open('src/data/apnea-ecg-database-1.0.0/list') as file_list:
            if not os.path.isdir('src/data/apnea-ecg-database-1.0.0/temp'):
                os.system('mkdir src/data/apnea-ecg-database-1.0.0/temp')
            try:
                system.os('rdann -h')
            except:
                print("Must install rdann from Physionet")
            for f in file_list:
                f = f.strip()
                if f != '\n' and f[0]!='x':
                    os.system('rdann -r src/data/apnea-ecg-database-1.0.0/{0} -a apn -f 0 > src/data/apnea-ecg-database-1.0.0/temp/{0}.txt'.format(f))
                    att, ident = wfdb.rdsamp('src/data/apnea-ecg-database-1.0.0/{0}'.format(f))
                    SIG_LEN = len(att)
                    print("Read: ", SIG_LEN, " values from ", f)
                    att = np.reshape(att, (SIG_LEN,1))
                    with open('src/data/apnea-ecg-database-1.0.0/temp/{}.txt'.format(f)) as g_list:
                        for g in g_list:
                            g = g.strip().split(' ')
                            g = [i for i in g if i != '']
                            #att, ident = wfdb.rdsamp('src/data/apnea-ecg-database-1.0.0/{0}'.format(f), sampfrom=int(g[1]), sampto=int(g[1])+5999, warn_empty=True)
                            if (int(g[1])+6000) < SIG_LEN:
                                att_file.write('{}\n'.format(','.join([str(i[0]) for i in resample(att[int(g[1]):int(g[1])+6000], 600)])))
                                lab_file.write('{}\n'.format(0 if g[2] == 'N' else 1))
                                #attributes = np.append(attributes, att[int(g[1]):int(g[1])+5999])
                                labels_clean = np.append(labels_clean, [0 if g[2] == 'N' else 1])
        labels_clean = np.array(labels_clean)
        att_file.close()
        lab_file.close()

        #Create label sets for BS1
        add_ncar(labels_clean, PATH + 'bs1_labels', 2)
        add_nar(labels_clean, PATH + 'bs1_labels', 2)
        add_nnar([], labels_clean, PATH + 'bs1_labels', 2, att_file=PATH+'bs1_attributes_train.csv')

        #Create Test Set
        attributes = []
        labels_clean = []
        os.system('rm {}'.format(PATH + 'bs1_attributes_test.csv'))
        os.system('rm {}'.format(PATH + 'bs1_labels_test.csv'))
        att_file = open(PATH + 'bs1_attributes_test.csv', 'a+')
        lab_file = open(PATH + 'bs1_labels_test.csv', 'a+')
        with open('src/data/apnea-ecg-database-1.0.0/list') as file_list:
            if not os.path.isdir('src/data/apnea-ecg-database-1.0.0/temp'):
                os.system('mkdir src/data/apnea-ecg-database-1.0.0/temp')
            try:
                system.os('rdann -h')
            except:
                print("Must install rdann from Physionet")
            for f in file_list:
                f = f.strip()
                if f != '\n' and f[0]=='x':
                    os.system('rdann -r src/data/apnea-ecg-database-1.0.0/{0} -a apn -f 0 > src/data/apnea-ecg-database-1.0.0/temp/{0}.txt'.format(f))
                    att, ident = wfdb.rdsamp('src/data/apnea-ecg-database-1.0.0/{0}'.format(f))
                    SIG_LEN = len(att)
                    print("Read: ", SIG_LEN, " values from ", f)
                    att = np.reshape(att, (SIG_LEN,1))
                    with open('src/data/apnea-ecg-database-1.0.0/temp/{}.txt'.format(f)) as g_list:
                        for g in g_list:
                            g = g.strip().split(' ')
                            g = [i for i in g if i != '']
                            #att, ident = wfdb.rdsamp('src/data/apnea-ecg-database-1.0.0/{0}'.format(f), sampfrom=int(g[1]), sampto=int(g[1])+5999, warn_empty=True)
                            if (int(g[1])+6000) < SIG_LEN:
                                att_file.write('{}\n'.format(','.join([str(i[0]) for i in resample(att[int(g[1]):int(g[1])+6000], 600)])))
                                lab_file.write('{}\n'.format(0 if g[2] == 'N' else 1))
                                #attributes = np.append(attributes, att[int(g[1]):int(g[1])+5999])
                                labels_clean = np.append(labels_clean, [0 if g[2] == 'N' else 1])
        att_file.close()
        lab_file.close()

        #Process PD Gait set into BioSignal Set 2
        #window the data to 5 second segments
        #I'm going to use Si as the test set, which has 64 walks
        #Ga has 113 and Ju has 129
        print("##### Preparing Dataset: BS2 #####")
        attributes = []
        labels_clean = []
        lab_test = []
        os.system('rm {}'.format(PATH + 'bs2_attributes.csv'))
        att_file = open(PATH + 'bs2_attributes_train.csv', 'a+')
        att_test = open(PATH + 'bs2_attributes_test.csv', 'a+')
        file_list = os.listdir('src/data/gait-in-parkinsons-disease-1.0.0')
        skip_file = [
            'SHA256SUMS.txt', 'gaitpd.png', 'demographics.html',
            'format.txt', 'demographics.xls', 'demographics.txt'
        ]
        counter = 0
        for f in file_list:
            if f in skip_file:
                continue
            #print(f)
            counter += 1
            with open('src/data/gait-in-parkinsons-disease-1.0.0/' + f) as gait_file:
                gait = gait_file.read()
                gait = gait.strip().split('\n')
                #print("Number of samples in gait", len(gait))
                for i in range(0, len(gait)-500, 500):
                    left_walk = []
                    right_walk = []
                    for j in range(i, i+500):
                        left_walk = np.append(left_walk, gait[j].split('\t')[17])
                        right_walk = np.append(right_walk, gait[j].split('\t')[18])

                    #attributes = np.append(attributes, left_walk)
                    #attributes = np.append(attributes, right_walk)
                    if 'Si' in f:
                        att_test.write('{}\n'.format(','.join([str(i) for i in left_walk])))
                        att_test.write('{}\n'.format(','.join([str(i) for i in right_walk])))
                        att_test.flush()
                        lab_test = np.append(lab_test, 0 if 'Co' in f else 1)
                    else:
                        att_file.write('{}\n'.format(','.join([str(i) for i in left_walk])))
                        att_file.write('{}\n'.format(','.join([str(i) for i in right_walk])))
                        att_file.flush()
                        labels_clean = np.append(labels_clean, 0 if 'Co' in f else 1)

        labels_clean = np.array(labels_clean)
        att_file.close()
        print("Number of files read: ", counter)
        print("Number of labels: ", len(labels_clean))
        print("Number of controls in gait dataset: ", np.count_nonzero(labels_clean==0))
        print("Number of PD in gait dataset: ", np.count_nonzero(labels_clean==1))
        np.savetxt(PATH + 'bs2_labels_clean.csv', labels_clean, delimiter=',', fmt='%d')
        np.savetxt(PATH + 'bs2_labels_test.csv', lab_test, delimiter=',', fmt='%d')
        att_test.close()
        att_file.close()


        #Create label sets for BS2
        add_ncar(labels_clean, PATH + 'bs2_labels', 2)
        add_nar(labels_clean, PATH + 'bs2_labels', 2)
        add_nnar([], labels_clean, PATH + 'bs2_labels', 2, att_file=PATH+'bs2_attributes.csv', num_channels=2)
