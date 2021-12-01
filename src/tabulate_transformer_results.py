#Author: Gentry Atkinson
#Organization: Texas State University
#Data: 1 Dec, 2021
#Process transformer results

import numpy

def read_file(filename):
    file = open(filename, 'r')
    results = [i.split(',') for i in  file.read().split('\n')]
    print(results)


if __name__ == '__main__':
    sets = ['har1', 'ss1']
    noise = ['clean', 'ncar5', 'ncar10', 'nar5', 'nar10', 'nnar5', 'nnar10']

    print('### Reading All Transformer Results ###\n')
    for s in sets:
        for n in noise:
            print ('Set: '+s, ' Noise Type: '+n)
            read_file('results/transformer/'+s+'_'+n+'_results.csv')
