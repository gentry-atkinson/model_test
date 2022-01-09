#Author: Gentry Atkinson
#Organization: Texas State University
#Data: 9 January, 2022
#Archive of no longer used code

"""
From process_all_data.py
This code wrote the old biosignal sets to an easily readable CSV
"""
if RUN_BS:
    print("##### Preparing Dataset: BS1 #####")
    """
    BioSignal Set 1
    Apnea-ECG Database: ECG signals collected at Phillips-University
    T Penzel, GB Moody, RG Mark, AL Goldberger, JH Peter.
    The Apnea-ECG Database. Computers in Cardiology 2000;27:255-258.
    2 classes
    1 channel
    6000 samples in every instance
    17,020 train instances
    17,243 test instances
    """
    #Process Sleep Apnea set into BioSignal Set 1
    #The instances get downsampled from 6000 to 1000 because otherwise it all
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
                            att_file.write('{}\n'.format(', '.join([str(i[0]) for i in resample(att[int(g[1]):int(g[1])+6000], 1000)])))
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
    os.system('rm {}'.format(PATH + 'bs1_labels_test_clean.csv'))
    att_file = open(PATH + 'bs1_attributes_test.csv', 'a+')
    lab_file = open(PATH + 'bs1_labels_test_clean.csv', 'a+')
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
                            att_file.write('{}\n'.format(', '.join([str(i[0]) for i in resample(att[int(g[1]):int(g[1])+6000], 1000)])))
                            lab_file.write('{}\n'.format(0 if g[2] == 'N' else 1))
                            #attributes = np.append(attributes, att[int(g[1]):int(g[1])+5999])
                            labels_clean = np.append(labels_clean, [0 if g[2] == 'N' else 1])
    att_file.close()
    lab_file.close()

    add_ncar(labels_clean, PATH + 'bs1_labels_test', 2)
    add_nar(labels_clean, PATH + 'bs1_labels_test', 2)
    add_nnar([], labels_clean, PATH + 'bs1_labels_test', 2, att_file=PATH+'bs1_attributes_test.csv')

    #Process PD Gait set into BioSignal Set 2
    #window the data to 10 second segments
    #I'm going to use Si as the test set, which has 64 walks
    #Ga has 113 and Ju has 129
    print("##### Preparing Dataset: BS2 #####")
    """
    BioSignal Set 2
    Gait in Parkinsons: measures of foot preassure while subjects walk
    Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & Stanley, H. E. (2000).
    PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals.
    Circulation [Online]. 101 (23), pp. e215–e220.
    2 classes
    2 channel
    1000 samples in every instance
    4757 train instances
    1472 test instances
    """
    attributes = []
    labels_clean = []
    lab_test = []
    os.system('rm {}'.format(PATH + 'bs2_attributes_train.csv'))
    os.system('rm {}'.format(PATH + 'bs2_attributes_test.csv'))
    att_file = open(PATH + 'bs2_attributes_train.csv', 'a+')
    att_test = open(PATH + 'bs2_attributes_test.csv', 'a+')
    file_list = os.listdir('src/data/gait-in-parkinsons-disease-1.0.0')
    file_list = shuffle(file_list, random_state=1899)
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
            for i in range(0, len(gait)-1000, 500):
                left_walk = []
                right_walk = []
                for j in range(i, i+1000):
                    left_walk = np.append(left_walk, gait[j].split('\t')[17])
                    right_walk = np.append(right_walk, gait[j].split('\t')[18])

                #attributes = np.append(attributes, left_walk)
                #attributes = np.append(attributes, right_walk)
                if 'Si' in f:
                    att_test.write('{}\n'.format(', '.join([str(i) for i in left_walk])))
                    att_test.write('{}\n'.format(', '.join([str(i) for i in right_walk])))
                    att_test.flush()
                    lab_test = np.append(lab_test, 0 if 'Co' in f else 1)
                else:
                    att_file.write('{}\n'.format(', '.join([str(i) for i in left_walk])))
                    att_file.write('{}\n'.format(', '.join([str(i) for i in right_walk])))
                    att_file.flush()
                    labels_clean = np.append(labels_clean, 0 if 'Co' in f else 1)

    labels_clean = np.array(labels_clean)
    att_file.close()
    print("Number of files read: ", counter)
    print("Number of labels: ", len(labels_clean))
    print("Number of controls in gait dataset: ", np.count_nonzero(labels_clean==0))
    print("Number of PD in gait dataset: ", np.count_nonzero(labels_clean==1))
    np.savetxt(PATH + 'bs2_labels_clean.csv', labels_clean, delimiter=', ', fmt='%d')
    np.savetxt(PATH + 'bs2_labels_test_clean.csv', lab_test, delimiter=', ', fmt='%d')
    att_test.close()
    att_file.close()


    #Create label sets for BS2
    add_ncar(labels_clean, PATH + 'bs2_labels', 2)
    add_nar(labels_clean, PATH + 'bs2_labels', 2)
    add_nnar([], labels_clean, PATH + 'bs2_labels', 2, att_file=PATH+'bs2_attributes_train.csv', num_channels=2)

    add_ncar(lab_test, PATH + 'bs2_labels_test', 2)
    add_nar(lab_test, PATH + 'bs2_labels_test', 2)
    add_nnar([], lab_test, PATH + 'bs2_labels_test', 2, att_file=PATH+'bs2_attributes_test.csv', num_channels=2)
    print("Done with BS")
