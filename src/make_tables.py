#Author: Gentry Atkinson
#Organization: Texas State University
#Data: 09 November, 2021
#Make some nice tables for the paper

import pandas as pd

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
    "Clean":[""],
    "NCAR 5%":[""],
    "NCAR 10%":[""],
    "NAR 5%":[""],
    "NAR 10%":[""],
    "NNAR 5%":[""],
    "NNAR 10%":[""]
}

rf_dic = {
    "Clean":[""],
    "NCAR 5%":[""],
    "NCAR 10%":[""],
    "NAR 5%":[""],
    "NAR 10%":[""],
    "NNAR 5%":[""],
    "NNAR 10%":[""]
}

"""
Table 1
Verticle
CNN TER for symetrical train/test pairs
"""
def tab1():
    tab = pd.DataFrame.from_dict({"Noise Type":cnn_dic.keys(), "TER":cnn_dic.values()})
    return tab.to_latex()

"""
Table 2
Horizontal
CNN TER for symetrical train/test pairs
"""
def tab2():
    tab = pd.DataFrame.from_dict(cnn_dic)
    return tab.to_latex()

if __name__ == "__main__":
    outfile = open("tables_latex.txt", 'w+')

    outfile.write('### Table 1 ###\n\n')
    outfile.write(tab1())
    outfile.write('\n######\n')

    outfile.write('### Table 2 ###\n\n')
    outfile.write(tab2())
    outfile.write('\n######\n')
