#Author: Gentry Atkinson
#Organization: Texas State University
#Data: 21 July, 2021
#Train and test a CNN on the 6 datasets with their many label sets

from matplotlib import pyplot as plt
import numpy as np

cnn_all_avg_aer = [
    0.3083,	0.3320,	0.3588,	0.3427,	0.3795,	0.3370,	0.3717,
    0.3288,	0.3515,	0.3753,	0.3637,	0.3968,	0.3568,	0.3927,
    0.3320,	0.3520,	0.3787,	0.3665,	0.4020,	0.3600,	0.3902,
    0.3182,	0.3403,	0.3637,	0.3478,	0.3758,	0.3440,	0.3727,
    0.3692,	0.3890,	0.4087,	0.3788,	0.3892,	0.3755,	0.3870,
    0.3308,	0.3532,	0.3762,	0.3635,	0.3922,	0.3565,	0.3860,
    0.3472,	0.3675,	0.3883,	0.3687,	0.3942,	0.3675,	0.3863
]

lstm_all_avg_aer = [
    0.3398,	0.3615,	0.3852,	0.3680,	0.3920,	0.3625,	0.3883,
    0.3398,	0.3602,	0.3835,	0.3710,	0.4038,	0.3682,	0.3972,
    0.3517,	0.3718,	0.3908,	0.3777,	0.4037,	0.3717,	0.3945,
    0.3480,	0.3693,	0.3905,	0.3722,	0.3967,	0.3670,	0.3900,
    0.3883,	0.4040,	0.4227,	0.3942,	0.4020,	0.3925,	0.3972,
    0.3627,	0.3822,	0.4030,	0.3887,	0.4070,	0.3815,	0.4025,
    0.3667,	0.3842,	0.4053,	0.3772,	0.3893,	0.3750,	0.3867
]

svm_all_avg_aer = [
    0.2992,	0.3213,	0.3480,	0.3283,	0.3570,	0.3220,	0.3483,
    0.2995,	0.3215,	0.3480,	0.3285,	0.3575,	0.3230,	0.3490,
    0.2968,	0.3190,	0.3457,	0.3282,	0.3578,	0.3212,	0.3512,
    0.3083,	0.3305,	0.3555,	0.3320,	0.3573,	0.3277,	0.3488,
    0.3373,	0.3582,	0.3820,	0.3520,	0.3667,	0.3472,	0.3588,
    0.3093,	0.3313,	0.3560,	0.3328,	0.3575,	0.3275,	0.3483,
    0.3295,	0.3505,	0.3752,	0.3435,	0.3575,	0.3373,	0.3488
]

nb_all_avg_aer = [
    0.3608,	0.3812,	0.4037,	0.3842,	0.4070,	0.3777,	0.4003,
    0.3522,	0.3735,	0.3952,	0.3738,	0.3942,	0.3673,	0.3885,
    0.3353,	0.3553,	0.3825,	0.3608,	0.3853,	0.3552,	0.3792,
    0.3720,	0.3918,	0.4130,	0.3927,	0.4152,	0.3868,	0.4080,
    0.3707,	0.3905,	0.4113,	0.3913,	0.4138,	0.3850,	0.4063,
    0.3653,	0.3852,	0.4067,	0.3883,	0.4120,	0.3818,	0.4052,
    0.3697,	0.3893,	0.4113,	0.3932,	0.4170,	0.3852,	0.4080
]

rf_all_avg_aer = [
    0.2930,	0.3183,	0.3460,	0.3243,	0.3535,	0.3163,	0.3455,
    0.2788,	0.3058,	0.3330,	0.3097,	0.3362,	0.3013,	0.3312,
    0.2805,	0.3053,	0.3337,	0.3108,	0.3387,	0.3038,	0.3332,
    0.3083,	0.3323,	0.3607,	0.3348,	0.3580,	0.3270,	0.3513,
    0.3173,	0.3425,	0.3662,	0.3325,	0.3472,	0.3278,	0.3388,
    0.2962,	0.3210,	0.3490,	0.3193,	0.3397,	0.3118,	0.3330,
    0.3413,	0.3630,	0.3868,	0.3568,	0.3707,	0.3485,	0.3615,
]

all_all_avg_aer = [(
    cnn_all_avg_aer[i]+ lstm_all_avg_aer[i]+ svm_all_avg_aer[i]+nb_all_avg_aer[i]+ rf_all_avg_aer[i])/5 for i in range(49)]



"""
Plot 1
Bar Plot
Clean, Avg, and Max AER for all models
Following: https://matplotlib.org/stable/gallery/misc/table_demo.html#sphx-glr-gallery-misc-table-demo-py
"""
def plot1():
    clean = [cnn_all_avg_aer[0], lstm_all_avg_aer[0],svm_all_avg_aer[0],nb_all_avg_aer[0],rf_all_avg_aer[0]]
    avgs = [sum(cnn_all_avg_aer)/49, sum(lstm_all_avg_aer)/49,sum(svm_all_avg_aer)/49,sum(nb_all_avg_aer)/49,sum(rf_all_avg_aer)/49]
    maxs = [max(cnn_all_avg_aer), max(lstm_all_avg_aer),max(svm_all_avg_aer),max(nb_all_avg_aer),max(rf_all_avg_aer)]

    n_cols = len(mins)
    cols = ["CNN", "LSTM", "SVM", "N. Bayes", "R. Forest"]
    colors =  plt.cm.BuPu(np.linspace(0.2, 0.4, 3))
    WIDTH = 0.8

    for i in range(n_cols):
        print('m:{} c:{} a:{} M:{}'.format(mins[i], clean[i], avgs[i], maxs[i]))
        #plt max bar
        plt.bar(cols[i], maxs[i], width=WIDTH, bottom=0, align='center', color=colors[2])
        # #plt avg bar
        plt.bar(cols[i], avgs[i], width=WIDTH, bottom=0, align='center', color=colors[1])
        #plt clean bar
        plt.bar(cols[i], clean[i], width=WIDTH, bottom=0, align='center', color=colors[0])


    ax = plt.gca()
    ax.set_ylim([0.2, 0.5])
    plt.title("Min/Avg/Max Error for Each Model")
    plt.savefig("imgs/plots/aer_for_all_models.pdf")

"""
Plot 2
Bar Plot
Min, Avg, and Max AER for all symetrical noise
"""
def plot2():
    data = [
        [cnn_all_avg_aer[0], lstm_all_avg_aer[0], svm_all_avg_aer[0], nb_all_avg_aer[0], rf_all_avg_aer[0]],
        [cnn_all_avg_aer[8], lstm_all_avg_aer[8], svm_all_avg_aer[8], nb_all_avg_aer[8], rf_all_avg_aer[8]],
        [cnn_all_avg_aer[16], lstm_all_avg_aer[16], svm_all_avg_aer[16], nb_all_avg_aer[16], rf_all_avg_aer[16]],
        [cnn_all_avg_aer[24], lstm_all_avg_aer[24], svm_all_avg_aer[24], nb_all_avg_aer[24], rf_all_avg_aer[24]],
        [cnn_all_avg_aer[32], lstm_all_avg_aer[32], svm_all_avg_aer[32], nb_all_avg_aer[32], rf_all_avg_aer[32]],
        [cnn_all_avg_aer[40], lstm_all_avg_aer[40], svm_all_avg_aer[40], nb_all_avg_aer[40], rf_all_avg_aer[40]],
        [cnn_all_avg_aer[48], lstm_all_avg_aer[48], svm_all_avg_aer[48], nb_all_avg_aer[48], rf_all_avg_aer[48]]
    ]

    n_cols = len(data)
    cols = ['Clean', 'NCAR05', 'NCAR10', 'NAR05', 'NAR10', 'NNAR05', 'NNAR10']
    colors =  plt.cm.BuPu(np.linspace(0.2, 0.4, 3))
    WIDTH = 0.8

    for i in range(n_cols):
        #plt max bar
        plt.bar(cols[i], max(data[i]), width=WIDTH, bottom=0, align='center', color=colors[2])
        # #plt avg bar
        plt.bar(cols[i], np.mean(data[i]), width=WIDTH, bottom=0, align='center', color=colors[1])
        #plt clean bar
        plt.bar(cols[i], min(data[i]), width=WIDTH, bottom=0, align='center', color=colors[0])

    ax = plt.gca()
    ax.set_ylim([0.2, 0.5])
    plt.title("Min/Avg/Max Error for Each Noise Class")
    plt.savefig("imgs/plots/aer_for_all_noise.pdf")

if __name__ == '__main__':
    plot1()
    plot2()
