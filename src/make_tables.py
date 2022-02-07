#Author: Gentry Atkinson
#Organization: Texas State University
#Data: 09 November, 2021
#Make some nice tables for the paper

import pandas as pd
import numpy as np
from model_config import loadDic

cnn_dic = {
    "Clean":["0.308"],
    "NCAR 5%":["0.335"],
    "NCAR 10%":["0.348"],
    "NAR 5%":["0.298-0.398"],
    "NAR 10%":["0.289-0.489"],
    "NNAR 5%":["0.307-0.407"],
    "NNAR 10%":["0.286-0.486"]
}

lstm_dic = {
    "Clean":["0.340"],
    "NCAR 5%":["0.345"],
    "NCAR 10%":["0.363"],
    "NAR 5%":["0.322-0.422"],
    "NAR 10%":["0.302-0.502"],
    "NNAR 5%":["0.332-0.432"],
    "NNAR 10%":["0.287-0.487"]
}

svm_dic = {
    "Clean":["0.299"],
    "NCAR 5%":["0.302"],
    "NCAR 10%":["0.307"],
    "NAR 5%":["0.282-0.382"],
    "NAR 10%":["0.267-0.467"],
    "NNAR 5%":["0.278-0.378"],
    "NNAR 10%":["0.249-0.449"]
}

nb_dic = {
    "Clean":["0.361"],
    "NCAR 5%":["0.359"],
    "NCAR 10%":["0.353"],
    "NAR 5%":["0.343-0.443"],
    "NAR 10%":["0.314-0.514"],
    "NNAR 5%":["0.332-0.432"],
    "NNAR 10%":["0.308-0.508"]
}

rf_dic = {
    "Clean":["0.293"],
    "NCAR 5%":["0.285"],
    "NCAR 10%":["0.292"],
    "NAR 5%":["0.285-0.385"],
    "NAR 10%":["0.247-0.447"],
    "NNAR 5%":["0.262-0.362"],
    "NNAR 10%":["0.261-0.461"]
}

all_dic = {
    "Clean":["0.320"],
    "NCAR 5%":["0.325"],
    "NCAR 10%":["0.333"],
    "NAR 5%":["0.306-0.406"],
    "NAR 10%":["0.284-0.484"],
    "NNAR 5%":["0.302-0.402"],
    "NNAR 10%":["0.278-0.478"]
}

train_test_comparison_dic = {
    'Model': ["CNN", "LSTM", "SVM", "N. Bayes", "R. Forest"],
    'Clean -> Noisy': [0.0453, 0.0364, 0.0383, 0.0315, 0.0410],
    'Noisy -> Clean': [0.0294, 0.0197, 0.0143, 0.0000, 0.0108],
    'Noisy -> Noisy': [0.0655, 0.0484, 0.0462, 0.0308, 0.0438]
}

dataset_dic = {
    'Name' : ['Synthetic 1', 'Synthtic 2', 'Apnea-ECG Database', 'Gait in Parkinsons', 'TX State Wrist HAR', 'UCI HAR'],
    'Short Title' : ['SS1', 'SS2', 'BS1', 'BS2', 'HAR1', 'HAR'],
    '# Classes' : [2, 5, 2, 2, 6, 6],
    '# Channels' : [1, 1, 1, 2, 1, 3],
    '# Train Instances' : [8000, 24000, 17020, 4757 ,2077, 7352],
    '# Train Instances' : [2000, 6000, 17243 , 1472 , 1091 , 2947]
}

#transformer avg aers
#different shape from alex's result
#row-> noise type
#col-> dataset
tran_all_aer = [
    [0.10, 0.28, 0.38, 0.57, 0.27, 0.29],
    [0.16, 0.34, 0.39, 0.10, 0.37, 0.31],
    [0.21, 0.39, 0.41, 0.56, 0.39, 0.36],
    [0.16, 0.33, 0.44, 0.51, 0.36, 0.37],
    [0.21, 0.38, 0.49, 0.46, 0.37, 0.41],
    [0.10, 0.30, 0.38, 0.54, 0.30, 0.37],
    [0.10, 0.31, 0.38, 0.47, 0.42, 0.44]
]

trans_dic = {
    "Clean":[str(np.mean(tran_all_aer[0]))],
    "NCAR 5%":[str((np.mean(tran_all_aer[1]) - 0.05)/(0.9))],
    "NCAR 10%":[str((np.mean(tran_all_aer[2]) - 0.1)/(0.8))],
    "NAR 5%":[str(np.mean(tran_all_aer[3])-0.05)+'-'+str(np.mean(tran_all_aer[3])+0.05)],
    "NAR 10%":[str(np.mean(tran_all_aer[4])-0.1)+'-'+str(np.mean(tran_all_aer[4])+0.1)],
    "NNAR 5%":[str(np.mean(tran_all_aer[5])-0.05)+'-'+str(np.mean(tran_all_aer[5])+0.05)],
    "NNAR 10%":[str(np.mean(tran_all_aer[6])-0.1)+'-'+str(np.mean(tran_all_aer[6])+0.1)]
}

def hor_tab(mod_dic):
    tab = pd.DataFrame.from_dict(mod_dic)
    return tab.to_latex(index=False)
def ver_tab(mod_dic):
    tab = pd.DataFrame.from_dict({"Noise Type":mod_dic.keys(), "TER":mod_dic.values()})
    return tab.to_latex(index=False)

def cnn_hyper_par():
    config_dic = loadDic('CNN')
    sets = ['ss1', 'ss2', 'sn1', 'sn2', 'har1', 'har2']
    hp_dic = {
        'Sets' : sets,
        'Layer 1 Filters' : [config_dic[s]['l1_numFilters'] for s in sets],
        'Layer 2 Filters' : [config_dic[s]['l2_numFilters'] for s in sets],
        'Layer 1 Kernel Size' : [config_dic[s]['l1_kernelSize'] for s in sets],
        'Layer 2 Kernel Size' : [config_dic[s]['l2_kernelSize'] for s in sets]
    }
    return hp_dic

def lstm_hyper_par():
    config_dic = loadDic('LSTM')
    sets = ['ss1', 'ss2', 'sn1', 'sn2', 'har1', 'har2']
    hp_dic = {
        'Sets' : sets,
        'Units' : [config_dic[s]['lstm_units'] for s in sets],
        'Hidden Layer Nodes' : [config_dic[s]['hidden_dense_size'] for s in sets]
    }
    return hp_dic

def transf_hyper_par():
    config_dic = loadDic('Transformer')
    sets = ['ss1', 'ss2', 'sn1', 'sn2', 'har1', 'har2']
    hp_dic = {
        'Sets' : sets,
        'Head Size' : [config_dic[s]['head_size'] for s in sets],
        'Number of Heads' : [config_dic[s]['num_heads'] for s in sets],
        'Feed Forward Filters' : [config_dic[s]['ff_dim'] for s in sets]
    }
    return hp_dic



if __name__ == "__main__":
    outfile = open("tables_latex.txt", 'w+')

    # outfile.write('### Table 1: CNN Horizontal ###\n\n')
    # outfile.write(hor_tab(cnn_dic))
    # outfile.write('\n######\n')

    # outfile.write('### Table 2: CNN Vertical ###\n\n')
    # outfile.write(ver_tab(cnn_dic))
    # outfile.write('\n######\n')

    # outfile.write('### Table 3: LSTM Horizontal ###\n\n')
    # outfile.write(hor_tab(lstm_dic))
    # outfile.write('\n######\n')

    # outfile.write('### Table 4: LSTM Vertical ###\n\n')
    # outfile.write(ver_tab(lstm_dic))
    # outfile.write('\n######\n')

    # outfile.write('### Table 5: SVM Horizontal ###\n\n')
    # outfile.write(hor_tab(svm_dic))
    # outfile.write('\n######\n')

    # outfile.write('### Table 6: SVM Vertical ###\n\n')
    # outfile.write(ver_tab(svm_dic))
    # outfile.write('\n######\n')

    # outfile.write('### Table 7: NB Horizontal ###\n\n')
    # outfile.write(hor_tab(nb_dic))
    # outfile.write('\n######\n')

    # outfile.write('### Table 8: NB Vertical ###\n\n')
    # outfile.write(ver_tab(nb_dic))
    # outfile.write('\n######\n')

    # outfile.write('### Table 9: RF Horizontal ###\n\n')
    # outfile.write(hor_tab(rf_dic))
    # outfile.write('\n######\n')

    # outfile.write('### Table 10: RF Vertical ###\n\n')
    # outfile.write(ver_tab(rf_dic))
    # outfile.write('\n######\n')

    # outfile.write('### Table 11: All Model Horizontal ###\n\n')
    # outfile.write(hor_tab(all_dic))
    # outfile.write('\n######\n')

    # outfile.write('### Table 12: All Model Vertical ###\n\n')
    # outfile.write(ver_tab(all_dic))
    # outfile.write('\n######\n')

    # #dics = [cnn_dic, lstm_dic, svm_dic, nb_dic, rf_dic]
    # big_table_dic = {
    #     "Noise Type":cnn_dic.keys(),
    #     "CNN":cnn_dic.values(),
    #     "LSTM":lstm_dic.values(),
    #     "Transformer":trans_dic.values(),
    #     "SVM":svm_dic.values(),
    #     "N. Bayes":nb_dic.values(),
    #     "R. Forest":rf_dic.values(),
    # }
    # outfile.write('### Table 13: TER big table ###\n\n')
    # outfile.write(pd.DataFrame.from_dict(big_table_dic).to_latex(index=False))
    # outfile.write('\n######\n')

    # outfile.write('### Table 14: Train -> Test Table ###\n\n')
    # outfile.write(pd.DataFrame.from_dict(train_test_comparison_dic).to_latex(index=False))
    # outfile.write('\n######\n')

    # outfile.write('### Table 15: Datasets ###\n\n')
    # outfile.write(pd.DataFrame.from_dict(dataset_dic).to_latex(index=False))
    # outfile.write('\n######\n')

    outfile.write('### Table 16: CNN hyperparameters ###\n\n')
    outfile.write(pd.DataFrame.from_dict(cnn_hyper_par()).to_latex(index=False))
    outfile.write('\n######\n')

    outfile.write('### Table 17: LSTM hyperparameters ###\n\n')
    outfile.write(pd.DataFrame.from_dict(lstm_hyper_par()).to_latex(index=False))
    outfile.write('\n######\n')

    outfile.write('### Table 18: Transformer hyperparameters ###\n\n')
    outfile.write(pd.DataFrame.from_dict(transf_hyper_par()).to_latex(index=False))
    outfile.write('\n######\n')
