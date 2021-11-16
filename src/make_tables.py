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


def hor_tab(mod_dic):
    tab = pd.DataFrame.from_dict(mod_dic)
    return tab.to_latex()
def ver_tab(mod_dic):
    tab = pd.DataFrame.from_dict({"Noise Type":mod_dic.keys(), "TER":mod_dic.values()})
    return tab.to_latex()



if __name__ == "__main__":
    outfile = open("tables_latex.txt", 'w+')

    outfile.write('### Table 1: CNN Horizontal ###\n\n')
    outfile.write(hor_tab(cnn_dic))
    outfile.write('\n######\n')

    outfile.write('### Table 2: CNN Vertical ###\n\n')
    outfile.write(ver_tab(cnn_dic))
    outfile.write('\n######\n')

    outfile.write('### Table 3: LSTM Horizontal ###\n\n')
    outfile.write(hor_tab(lstm_dic))
    outfile.write('\n######\n')

    outfile.write('### Table 4: LSTM Vertical ###\n\n')
    outfile.write(ver_tab(lstm_dic))
    outfile.write('\n######\n')

    outfile.write('### Table 5: SVM Horizontal ###\n\n')
    outfile.write(hor_tab(svm_dic))
    outfile.write('\n######\n')

    outfile.write('### Table 6: SVM Vertical ###\n\n')
    outfile.write(ver_tab(svm_dic))
    outfile.write('\n######\n')

    outfile.write('### Table 7: NB Horizontal ###\n\n')
    outfile.write(hor_tab(nb_dic))
    outfile.write('\n######\n')

    outfile.write('### Table 8: NB Vertical ###\n\n')
    outfile.write(ver_tab(nb_dic))
    outfile.write('\n######\n')

    outfile.write('### Table 9: RF Horizontal ###\n\n')
    outfile.write(hor_tab(rf_dic))
    outfile.write('\n######\n')

    outfile.write('### Table 10: RF Vertical ###\n\n')
    outfile.write(ver_tab(rf_dic))
    outfile.write('\n######\n')
