#Author: Gentry Atkinson
#Organization: Texas State University
#Data: 09 November, 2021
#Make some nice tables for the paper

import pandas as pd

cnn_dic = {
    "Clean":"0.308",
    "NCAR 5%":"0.335",
    "NCAR 10%":"0.348",
    "NAR 5%":"0.298-0.398",
    "NAR 10%":"0.289-0.489",
    "NNAR 5%":"0.307-0.407",
    "NNAR 10%":"0.286-0.486"
}

def tab1():
    print(pd.DataFrame.from_dict(cnn_dic))

if __name__ == "__main__":
    tab1()
