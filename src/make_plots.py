#Author: Gentry Atkinson
#Organization: Texas State University
#Data: 09 November, 2021
#Make some nice pyplots for the paper

from matplotlib import pyplot as plt
import numpy as np
import seaborn as sb

LABEL_TILT = 25

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

all_model_ss1_avg_aer = [
    0.1076,	0.1444,	0.1874,	0.1436,	0.1852,	0.1238,	0.1594,
    0.1192,	0.1544,	0.1942,	0.1566,	0.2004,	0.1410,	0.1770,
    0.1316,	0.1654,	0.2042,	0.1686,	0.2116,	0.1492,	0.1838,
    0.1192,	0.1542,	0.1950,	0.1512,	0.1904,	0.1336,	0.1656,
    0.1310,	0.1636,	0.2038,	0.1614,	0.1974,	0.1424,	0.1678,
    0.1186,	0.1546,	0.1942,	0.1536,	0.1860,	0.1274,	0.1574,
    0.1284,	0.1600,	0.2062,	0.1594,	0.1944,	0.1344,	0.1594
]

all_model_ss2_avg_aer = [
    0.3108,	0.3380,	0.3714,	0.3406,	0.3688,	0.3378,	0.3626,
    0.3276,	0.3540,	0.3856,	0.3528,	0.3750,	0.3504,	0.3712,
    0.3238,	0.3496,	0.3814,	0.3530,	0.3796,	0.3500,	0.3740,
    0.3258,	0.3518,	0.3848,	0.3550,	0.3826,	0.3524,	0.3776,
    0.3574,	0.3816,	0.4118,	0.3880,	0.4180,	0.3850,	0.4116,
    0.3192,	0.3448,	0.3788,	0.3504,	0.3792,	0.3474,	0.3740,
    0.3410,	0.3654,	0.3974,	0.3720,	0.4004,	0.3690,	0.3944
]

all_model_bs1_avg_aer = [
    0.4602,	0.4652,	0.4678,	0.4642,	0.4682,	0.4634,	0.4646,
    0.4632,	0.4678,	0.4696,	0.4688,	0.4734,	0.4668,	0.4702,
    0.4626,	0.4672,	0.4698,	0.4670,	0.4724,	0.4666,	0.4704,
    0.4720,	0.4760,	0.4772,	0.4694,	0.4666,	0.4694,	0.4648,
    0.4756,	0.4804,	0.4792,	0.4692,	0.4614,	0.4694,	0.4606,
    0.4662,	0.4704,	0.4722,	0.4664,	0.4646,	0.4658,	0.4626,
    0.4756,	0.4796,	0.4800,	0.4702,	0.4628,	0.4708,	0.4624
]

all_model_bs2_avg_aer = [
    0.4702,	0.4684,	0.4864,	0.5068,	0.5438,	0.5066,	0.5434,
    0.4732,	0.4722,	0.4874,	0.5092,	0.5472,	0.5072,	0.5484,
    0.4776,	0.4750,	0.4908,	0.5100,	0.5462,	0.5108,	0.5396,
    0.4688,	0.4658,	0.4840,	0.5042,	0.5406,	0.5022,	0.5374,
    0.4736,	0.4716,	0.4862,	0.5076,	0.5416,	0.5064,	0.5394,
    0.4808,	0.4784,	0.4934,	0.5174,	0.5510,	0.5156,	0.5516,
    0.4762,	0.4744,	0.4864,	0.5066,	0.5404,	0.5064,	0.5388
]

all_model_har1_avg_aer = [
    0.3036,	0.3350,	0.3608,	0.3380,	0.3632,	0.3242,	0.3590,
    0.2606,	0.2948,	0.3246,	0.2964,	0.3222,	0.2840,	0.3184,
    0.2360,	0.2682,	0.3022,	0.2720,	0.2966,	0.2576,	0.2942,
    0.3144,	0.3474,	0.3694,	0.3388,	0.3568,	0.3306,	0.3554,
    0.3236,	0.3532,	0.3750,	0.3342,	0.3412,	0.3300,	0.3418,
    0.3152,	0.3458,	0.3694,	0.3382,	0.3588,	0.3322,	0.3560,
    0.3474,	0.3768,	0.3948,	0.3594,	0.3696,	0.3556,	0.3690
]

all_model_har2_avg_aer = [
    0.2910,	0.3260,	0.3580,	0.3230,	0.3550,	0.3220,	0.3530,
    0.2900,	0.3250,	0.3540,	0.3200,	0.3510,	0.3200,	0.3470,
    0.2870,	0.3210,	0.3550,	0.3200,	0.3540,	0.3180,	0.3490,
    0.3080,	0.3420,	0.3720,	0.3320,	0.3530,	0.3280,	0.3510,
    0.3620,	0.3950,	0.4210,	0.3530,	0.3530,	0.3540,	0.3480,
    0.3220,	0.3550,	0.3860,	0.3410,	0.3540,	0.3370,	0.3530,
    0.3630,	0.3940,	0.4180,	0.3530,	0.3420,	0.3530,	0.3400
]

all_all_avg_aer_2 = [(
    all_model_ss1_avg_aer [i]+all_model_ss2_avg_aer [i]+ all_model_bs1_avg_aer [i]+ all_model_bs2_avg_aer [i]+all_model_har1_avg_aer [i]+ all_model_har2_avg_aer [i])/6 for i in range(49)]

#   Model[0] -> CEV values
#   Model[1] -> SDE values
bias_ss1_dic = {
    'CNN':[[1.240, 0.954, 0.001, 0.072, 0.093, 0.025],
            [1.575, 1.381, 0.035, 0.379, 0.432, 0.225]
        ],
    'LSTM': [[0.096, 0.149, 0.009, 0.507, 0.046, 0.096],
            [0.439, 0.546, 0.134, 1.007, 0.304, 0.438]
        ],
    'SVM': [[0.003, 0.051, 0.772, 2.975, 0.256, 1.557],
            [0.080, 0.320, 1.243, 2.439, 0.716, 1.765]
        ],
    'NB': [[0.591, 2.063, 1.768, 6.273, 1.390, 4.001],
            [1.087, 2.031, 1.880, 3.542, 1.667, 2.829]
        ],
    'RF': [[0.131, 0.432, 0.263, 0.328, 0.164, 0.084],
            [0.512, 0.930, 0.726, 0.810, 0.573, 0.410]
        ]
}

bias_ss2_dic = {
    'CNN':[[0.048, 0.092, 0.147, 0.686, 0.218, 0.763],
            [0.202, 0.279, 0.430, 0.911, 0.548, 1.070]
        ],
    'LSTM': [[0.161, 0.447, 0.274, 0.849, 0.130, 0.481],
            [0.390, 0.685, 0.555, 0.986, 0.485, 0.782]
        ],
    'SVM': [[0.367, 1.623, 0.710, 14.183, 0.800, 2.699],
            [0.219, 0.415, 0.668, 3.009, 0.724, 1.422]
        ],
    'NB': [[3.704, 1.702, 0.284, 0.602, 0.442, 1.141],
            [2.182, 0.670, 0.470, 0.651, 0.620, 1.037]
        ],
    'RF': [[0.198, 0.961, 1.476, 12.703, 1.286, 9.326],
            [0.251, 0.356, 1.022, 2.853, 1.009, 2.492]
        ]
}

bias_bs1_dic = {
    'CNN':[[0.009, 0.002, 0.136, 0.220, 0.027, 0.102],
            [0.131, 0.069, 0.521, 0.664, 0.233, 0.452]
        ],
    'LSTM': [[0.043, 0.000, 0.006, 0.025, 0.003, 0.041],
            [0.293, 0.030, 0.114, 0.224, 0.083, 0.287]
        ],
    'SVM': [[0.007, 0.030, 0.089, 0.235, 0.081, 0.212],
            [0.117, 0.245, 0.422, 0.685, 0.403, 0.651]
        ],
    'NB': [[0.052, 0.048, 0.041, 0.017, 0.001, 0.026],
            [0.322, 0.311, 0.285, 0.185, 0.053, 0.230]
        ],
    'RF': [[0.000, 0.010, 0.076, 0.372, 0.041, 0.261],
            [0.004, 0.139, 0.390, 0.863, 0.288, 0.723]
        ]
}

bias_bs2_dic = {
    'CNN':[[14.123, 5.190, 1.216, 4.678, 0.126, 9.111],
            [5.315, 3.222, 1.559, 3.059, 0.502, 4.269]
        ],
    'LSTM': [[0.692, 0.160, 0.001, 0.000, 0.019, 0.047],
            [1.176, 0.566, 0.038, 0.025, 0.196, 0.307]
        ],
    'SVM': [[], #Values missing
            []  #Values missing
        ],
    'NB': [[0.487, 1.237, 0.005, 0.026, 0.102, 0.024],
            [0.987, 1.573, 0.096, 0.226, 0.452, 0.219]
        ],
    'RF': [[0.001, 0.000, 0.000, 0.037, 0.002, 0.022],
            [0.041, 0.018, 0.031, 0.273, 0.057, 0.210]
        ]
}

bias_har1_dic = {
    'CNN':[[0.121, 3.097, 0.906, 4.672, 1.144, 2.336],
            [0.270, 1.305, 1.087, 1.776, 0.816, 1.326]
        ],
    'LSTM': [[0.254, 1.194, 0.373, 2.738, 0.201, 3.162],
            [0.581, 1.001, 0.437, 1.612, 0.323, 1.646]
        ],
    'SVM': [[], #Values missing
            []  #Values missing
        ],
    'NB': [[],  #Values missing
            []  #Values missing
        ],
    'RF': [[1.516, 4.354, 0.656, 5.701, 2.978, 3.028],
            [1.056, 1.851, 0.586, 2.164, 1.600, 1.642]
        ]
}

bias_har2_dic = {
    'CNN':[[8.764, 33.409, 81.898, 1644.171, 71.722, 296.200],
            [1.642, 0.932, 3.860, 17.439, 1.499, 1.513]
        ],
    'LSTM': [[],    #Values missing
            []      #Values missing
        ],
    'SVM': [[0.172, 0.673, 0.349, 11.237, 0.263, 10.413],
            [0.298, 0.614, 0.479, 2.077, 0.412, 1.925]
        ],
    'NB': [[0.428, 2.858, 0.479, 1.257, 0.585, 2.139],
            [0.557, 1.309, 0.648, 1.087, 0.713, 1.332]
        ],
    'RF': [[1.790, 4.224, 0.288, 2.614, 0.498, 2.108],
            [0.777, 1.075, 0.163, 0.897, 0.323, 0.738]
        ],
}





"""
Plot 1
Bar Plot
Clean, Avg, and Max AER for all models
Following: https://matplotlib.org/stable/gallery/misc/table_demo.html#sphx-glr-gallery-misc-table-demo-py
"""
def plot1():
    plt.figure()
    clean = [cnn_all_avg_aer[0], lstm_all_avg_aer[0],svm_all_avg_aer[0],nb_all_avg_aer[0],rf_all_avg_aer[0]]
    avgs = [sum(cnn_all_avg_aer)/49, sum(lstm_all_avg_aer)/49,sum(svm_all_avg_aer)/49,sum(nb_all_avg_aer)/49,sum(rf_all_avg_aer)/49]
    maxs = [max(cnn_all_avg_aer), max(lstm_all_avg_aer),max(svm_all_avg_aer),max(nb_all_avg_aer),max(rf_all_avg_aer)]

    n_cols = len(clean)
    cols = ["CNN", "LSTM", "SVM", "N. Bayes", "R. Forest"]
    colors =  plt.cm.Blues(np.linspace(0.3, 0.8, 3))
    WIDTH = 0.4

    for i in range(n_cols):
        #plt max bar
        plt.bar(cols[i], maxs[i], width=WIDTH, bottom=0, align='center', color=colors[2])
        # #plt avg bar
        plt.bar(cols[i], avgs[i], width=WIDTH, bottom=0, align='center', color=colors[1])
        #plt clean bar
        plt.bar(cols[i], clean[i], width=WIDTH, bottom=0, align='center', color=colors[0])


    ax = plt.gca()
    ax.set_ylim([0.1, 0.6])
    plt.xticks(rotation=LABEL_TILT)
    plt.title("Min/Avg/Max Error for Each Model")
    plt.savefig("imgs/plots/aer_for_all_models.pdf", bbox_inches='tight')

"""
Plot 2
Bar Plot
Min, Avg, and Max AER for all symetrical noise
"""
def plot2():
    plt.figure()
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
    colors =  plt.cm.Blues(np.linspace(0.3, 0.8, 3))
    WIDTH = 0.4

    for i in range(n_cols):
        #plt max bar
        plt.bar(cols[i], max(data[i]), width=WIDTH, bottom=0, align='center', color=colors[2])
        # #plt avg bar
        plt.bar(cols[i], np.mean(data[i]), width=WIDTH, bottom=0, align='center', color=colors[1])
        #plt clean bar
        plt.bar(cols[i], min(data[i]), width=WIDTH, bottom=0, align='center', color=colors[0])

    ax = plt.gca()
    ax.set_ylim([0.1, 0.6])
    plt.xticks(rotation=LABEL_TILT)
    plt.title("Min/Avg/Max Error for Each Noise Class")
    plt.savefig("imgs/plots/aer_for_all_noise.pdf", bbox_inches='tight')

def plotHeatMap(data, title, filename):
    cols = ['Clean', 'NCAR05', 'NCAR10', 'NAR05', 'NAR10', 'NNAR05', 'NNAR10']
    ax = sb.heatmap(data, annot=True,  cmap="YlGnBu", xticklabels=cols, yticklabels=cols)
    plt.xticks(rotation=LABEL_TILT)
    plt.title(title)
    plt.savefig(filename, bbox_inches='tight')

"""
Plot 3
Heat Map
Average AER for all train/test pairs
"""
def plot3():
    plt.figure()
    data = np.reshape(np.array(all_all_avg_aer), (7, 7))
    title = "All Model Avg AER for Each Train/Test"
    filename = "imgs/plots/all_aer_for_test_train_pairs.pdf"
    plotHeatMap(data, title, filename)

"""
Plot 4
Heat Map
CNN AER for all train/test pairs
"""
def plot4():
    plt.figure()
    data = np.reshape(np.array(cnn_all_avg_aer), (7, 7))
    title = "CNN AER for Each Train/Test"
    filename = "imgs/plots/cnn_aer_for_test_train_pairs.pdf"
    plotHeatMap(data, title, filename)



"""
Plot 5
Heat Map
LSTM AER for all train/test pairs
"""
def plot5():
    plt.figure()
    data = np.reshape(np.array(lstm_all_avg_aer), (7, 7))
    title = "LSTM AER for Each Train/Test"
    filename = "imgs/plots/lstm_aer_for_test_train_pairs.pdf"
    plotHeatMap(data, title, filename)

"""
Plot 6
Heat Map
SVM AER for all train/test pairs
"""
def plot6():
    plt.figure()
    data = np.reshape(np.array(svm_all_avg_aer), (7, 7))
    title = "SVM AER for Each Train/Test"
    filename = "imgs/plots/svm_aer_for_test_train_pairs.pdf"
    plotHeatMap(data, title, filename)

"""
Plot 7
Heat Map
Naive Bayes AER for all train/test pairs
"""
def plot7():
    plt.figure()
    data = np.reshape(np.array(svm_all_avg_aer), (7, 7))
    title = "N.Bayes AER for Each Train/Test"
    filename = "imgs/plots/nb_aer_for_test_train_pairs.pdf"
    plotHeatMap(data, title, filename)

"""
Plot 8
Heat Map
Naive Bayes AER for all train/test pairs
"""
def plot8():
    plt.figure()
    data = np.reshape(np.array(svm_all_avg_aer), (7, 7))
    title = "R. Forest AER for Each Train/Test"
    filename = "imgs/plots/rf_aer_for_test_train_pairs.pdf"
    plotHeatMap(data, title, filename)

"""
Plot 9
Heat Map
All model delta AER for all train/test pairs
"""
def plot9():
    plt.figure()
    data = np.reshape(np.array([((all_all_avg_aer[i] - all_all_avg_aer[0])) for i in range(49)]), (7, 7))
    title = "Change in AER for Each Train/Test"
    filename = "imgs/plots/delta_aer_for_test_train_pairs.pdf"
    plotHeatMap(data, title, filename)

"""
Plot 10
Bar Plot
Min, Avg, Max AER for each dataset
"""
def plot10():
    plt.figure()
    data = [
        all_model_ss1_avg_aer, all_model_ss2_avg_aer,
        all_model_bs1_avg_aer, all_model_bs2_avg_aer,
        all_model_har1_avg_aer, all_model_har2_avg_aer,
    ]

    n_cols = len(data)
    cols = ['Synthetic 1', 'Synthetic 2', 'Apnea-ECG', 'Gait Parkinsons', 'TXState HAR', 'UCI HAR']
    colors =  plt.cm.Blues(np.linspace(0.3, 0.8, 3))
    WIDTH = 0.4

    for i in range(n_cols):
        #plt max bar
        plt.bar(cols[i], max(data[i]), width=WIDTH, bottom=0, align='center', color=colors[2])
        # #plt avg bar
        plt.bar(cols[i], np.mean(data[i]), width=WIDTH, bottom=0, align='center', color=colors[1])
        #plt clean bar
        plt.bar(cols[i], min(data[i]), width=WIDTH, bottom=0, align='center', color=colors[0])

    ax = plt.gca()
    ax.set_ylim([0.1, 0.6])
    plt.xticks(rotation=LABEL_TILT)
    plt.title("Min/Avg/Max Error for Each Dataset")
    plt.savefig("imgs/plots/aer_for_all_datasets.pdf", bbox_inches='tight')

"""
Plot 11
Bar Plot
Min, Avg, Max Delta AER for each dataset
"""
def plot11():
    plt.figure()
    data = [
        [all_model_ss1_avg_aer[i] - all_model_ss1_avg_aer[0] for i in range(1, 49)],
        [all_model_ss2_avg_aer[i] - all_model_ss2_avg_aer[0] for i in range(1, 49)],
        [all_model_bs1_avg_aer[i] - all_model_bs1_avg_aer[0] for i in range(1, 49)],
        [all_model_bs2_avg_aer[i] - all_model_bs2_avg_aer[0] for i in range(1, 49)],
        [all_model_har1_avg_aer[i] - all_model_har1_avg_aer[0] for i in range(1, 49)],
        [all_model_har2_avg_aer[i] - all_model_har2_avg_aer[0] for i in range(1, 49)],
    ]

    n_cols = len(data)
    cols = ['Synthetic 1', 'Synthetic 2', 'Apnea-ECG', 'Gait Parkinsons', 'TXState HAR', 'UCI HAR']
    colors =  plt.cm.Blues(np.linspace(0.3, 0.8, 3))
    WIDTH = 0.4

    for i in range(n_cols):
        #plt max bar
        plt.bar(cols[i], max(data[i]), width=WIDTH, bottom=0, align='center', color=colors[2])
        # #plt avg bar
        plt.bar(cols[i], np.mean(data[i]), width=WIDTH, bottom=0, align='center', color=colors[1])
        #plt clean bar
        plt.bar(cols[i], min(data[i]), width=WIDTH, bottom=0, align='center', color=colors[0])

    ax = plt.gca()
    ax.set_ylim([0.0, 0.15])
    plt.xticks(rotation=LABEL_TILT)
    plt.title("Min/Avg/Max Error for Each Dataset")
    plt.savefig("imgs/plots/delta_aer_for_all_datasets.pdf", bbox_inches='tight')

"""
Plot 12
Bar Plot
Min, Avg, Max Delta AER for each model
"""
def plot12():
    plt.figure()
    data = [
        [cnn_all_avg_aer[i] - cnn_all_avg_aer[0] for i in range(1, 49)],
        [lstm_all_avg_aer[i] - cnn_all_avg_aer[0] for i in range(1, 49)],
        [svm_all_avg_aer[i] - cnn_all_avg_aer[0] for i in range(1, 49)],
        [nb_all_avg_aer[i] - cnn_all_avg_aer[0] for i in range(1, 49)],
        [rf_all_avg_aer[i] - cnn_all_avg_aer[0] for i in range(1, 49)],
    ]

    n_cols = len(data)
    cols = ["CNN", "LSTM", "SVM", "N. Bayes", "R. Forest"]
    colors =  plt.cm.Blues(np.linspace(0.3, 0.8, 3))
    WIDTH = 0.4

    for i in range(n_cols):
        #plt max bar
        plt.bar(cols[i], max(data[i]), width=WIDTH, bottom=0, align='center', color=colors[2])
        # #plt avg bar
        plt.bar(cols[i], np.mean(data[i]), width=WIDTH, bottom=0, align='center', color=colors[1])
        #plt clean bar
        plt.bar(cols[i], min(data[i]), width=WIDTH, bottom=0, align='center', color=colors[0])

    ax = plt.gca()
    ax.set_ylim([0.0, 0.15])
    plt.xticks(rotation=LABEL_TILT)
    plt.title("Min/Avg/Max Error for Each Model")
    plt.savefig("imgs/plots/delta_aer_for_all_models.pdf", bbox_inches='tight')

"""
Plot 13
Bar Plot
Min, Avg, Max Delta AER/% noise for each noise type
TODO
"""
def plot13():
    plt.figure()
    deltas = [
        [all_model_ss1_avg_aer[i] - all_model_ss1_avg_aer[0] for i in range(0, 49)],
        [all_model_ss2_avg_aer[i] - all_model_ss2_avg_aer[0] for i in range(0, 49)],
        [all_model_bs1_avg_aer[i] - all_model_bs1_avg_aer[0] for i in range(0, 49)],
        [all_model_bs2_avg_aer[i] - all_model_bs2_avg_aer[0] for i in range(0, 49)],
        [all_model_har1_avg_aer[i] - all_model_har1_avg_aer[0] for i in range(0, 49)],
        [all_model_har2_avg_aer[i] - all_model_har2_avg_aer[0] for i in range(0, 49)],
    ]

    divisors = [
        1, 5, 10, 5, 10, 5, 10,
        1, 5, 10, 5, 10, 5, 10,
        1, 5, 10, 5, 10, 5, 10,
        1, 5, 10, 5, 10, 5, 10,
        1, 5, 10, 5, 10, 5, 10,
        1, 5, 10, 5, 10, 5, 10,
        1, 5, 10, 5, 10, 5, 10
    ]

    deltas = np.array([np.array(i)/np.array(divisors) for i in deltas])
    #print(np.append(deltas[:, range(1, 49, 7)], deltas[:, range(2, 49, 7)]))
    data = [
        #ncar
        np.reshape([deltas[:,8], deltas[:,9], deltas[:,15], deltas[:,16]], (24)),
        #nar
        np.reshape([deltas[:,24], deltas[:,25], deltas[:,31], deltas[:,32]], (24)),
        #nnar
        np.reshape([deltas[:,40], deltas[:,41], deltas[:,47], deltas[:,48]], (24))
    ]
    n_cols = len(data)
    cols = ['NCAR', 'NAR', 'NNAR']
    colors =  plt.cm.Blues(np.linspace(0.3, 0.8, 3))
    WIDTH = 0.8
    for i in range(n_cols):
        #plt max bar
        #print(max(data[i]))
        plt.bar(cols[i], max(data[i]), width=WIDTH, bottom=0, align='center', color=colors[2])
        # #plt avg bar
        #print(np.mean(data[i]))
        plt.bar(cols[i], np.mean(data[i]), width=WIDTH, bottom=0, align='center', color=colors[1])
        #plt clean bar
        #print(min(data[i]))
        plt.bar(cols[i], min(data[i]), width=WIDTH, bottom=0, align='center', color=colors[0])
    plt.axhline(y=0.0, color='black', linestyle='-')

    ax = plt.gca()
    ax.set_ylim([-0.01, 0.02])
    plt.xticks(rotation=LABEL_TILT)
    plt.title("Min/Avg/Max Change Error per 1% Label Noise")
    plt.savefig("imgs/plots/delta_aer_per_percent_for_all_noises.pdf", bbox_inches='tight')

"""
Plot 14
Horizontal Bar Plot
Clean, Avg, and Max AER for all models
Following: https://matplotlib.org/stable/gallery/misc/table_demo.html#sphx-glr-gallery-misc-table-demo-py
"""
def plot14():
    plt.figure()
    clean = [cnn_all_avg_aer[0], lstm_all_avg_aer[0],svm_all_avg_aer[0],nb_all_avg_aer[0],rf_all_avg_aer[0]]
    avgs = [sum(cnn_all_avg_aer)/49, sum(lstm_all_avg_aer)/49,sum(svm_all_avg_aer)/49,sum(nb_all_avg_aer)/49,sum(rf_all_avg_aer)/49]
    maxs = [max(cnn_all_avg_aer), max(lstm_all_avg_aer),max(svm_all_avg_aer),max(nb_all_avg_aer),max(rf_all_avg_aer)]

    n_cols = len(clean)
    cols = ["CNN", "LSTM", "SVM", "N. Bayes", "R. Forest"]
    colors =  plt.cm.Blues(np.linspace(0.3, 0.8, 3))
    WIDTH = 0.4

    for i in range(n_cols):
        #plt max bar
        plt.barh(i, maxs[i], height=WIDTH, color=colors[2])
        # #plt avg bar
        plt.barh(i, avgs[i], height=WIDTH, color=colors[1])
        #plt clean bar
        plt.barh(i, clean[i], height=WIDTH, color=colors[0])


    ax = plt.gca()
    ax.set_yticks(range(len(cols)))
    ax.set_yticklabels(cols)
    plt.xticks(rotation=LABEL_TILT)
    ax.set_xlim([0.1, 0.6])
    plt.title("Min/Avg/Max Error for Each Model")
    plt.savefig("imgs/plots/aer_for_all_models_horizontal.pdf", bbox_inches='tight')

"""
Plot 15
Scatter Plot
CEV and SDE for all models on SS1
"""
def plot15():
    plt.figure()
    models = ["CNN", "LSTM", "SVM", "NB", "RF"]
    for i in range(len(models)):
        plt.scatter(bias_ss1_dic[models[i]][0], bias_ss1_dic[models[i]][1], label=models[i])
    plt.title("CEV/SDE")
    plt.legend()
    plt.savefig("imgs/plots/cev_and_sde_on_ss1.pdf", bbox_inches='tight')

"""
Plot 16
Scatter Plot
CEV and SDE for all models on SS2
"""
def plot16():
    plt.figure()
    models = ["CNN", "LSTM", "SVM", "NB", "RF"]
    for i in range(len(models)):
        plt.scatter(bias_ss2_dic[models[i]][0], bias_ss2_dic[models[i]][1], label=models[i])
    plt.title("CEV/SDE")
    plt.legend()
    plt.savefig("imgs/plots/cev_and_sde_on_ss2.pdf", bbox_inches='tight')

"""
Plot 17
Scatter Plot
CEV and SDE for all models on BS1
"""
def plot17():
    plt.figure()
    models = ["CNN", "LSTM", "SVM", "NB", "RF"]
    for i in range(len(models)):
        plt.scatter(bias_bs1_dic[models[i]][0], bias_bs1_dic[models[i]][1], label=models[i])
    plt.title("CEV/SDE")
    plt.legend()
    plt.savefig("imgs/plots/cev_and_sde_on_bs1.pdf", bbox_inches='tight')

"""
Plot 18
Scatter Plot
CEV and SDE for all models on BS2
"""
def plot18():
    plt.figure()
    models = ["CNN", "LSTM", "SVM", "NB", "RF"]
    for i in range(len(models)):
        plt.scatter(bias_bs2_dic[models[i]][0], bias_bs2_dic[models[i]][1], label=models[i])
    plt.title("CEV/SDE")
    plt.legend()
    plt.savefig("imgs/plots/cev_and_sde_on_bs2.pdf", bbox_inches='tight')

"""
Plot 19
Scatter Plot
CEV and SDE for all models on HAR1
"""
def plot19():
    plt.figure()
    models = ["CNN", "LSTM", "SVM", "NB", "RF"]
    for i in range(len(models)):
        plt.scatter(bias_har1_dic[models[i]][0], bias_har1_dic[models[i]][1], label=models[i])
    plt.title("CEV/SDE")
    plt.legend()
    plt.savefig("imgs/plots/cev_and_sde_on_har1.pdf", bbox_inches='tight')

"""
Plot 20
Scatter Plot
CEV and SDE for all models on HAR2
"""
def plot20():
    plt.figure()
    models = ["CNN", "LSTM", "SVM", "NB", "RF"]
    for i in range(len(models)):
        plt.scatter(bias_har2_dic[models[i]][0], bias_har2_dic[models[i]][1], label=models[i])
    plt.title("CEV/SDE")
    plt.legend()
    plt.savefig("imgs/plots/cev_and_sde_on_har2.pdf", bbox_inches='tight')

def plot21_temp():
    plt.figure()
    models = ["CNN", "LSTM", "SVM", "NB", "RF"]
    for i in range(len(models)):
        plt.scatter(bias_ss1_dic[models[i]][0], bias_ss1_dic[models[i]][1], label=models[i])
    for i in range(len(models)):
        plt.scatter(bias_bs1_dic[models[i]][0], bias_bs1_dic[models[i]][1], label=models[i])
    for i in range(len(models)):
        plt.scatter(bias_bs2_dic[models[i]][0], bias_bs2_dic[models[i]][1], label=models[i])
    plt.title("CEV/SDE")
    plt.savefig("imgs/plots/cev_and_sde_on_all_binayry_datasets.pdf", bbox_inches='tight')


if __name__ == '__main__':
    plt.rcParams.update({'font.size': 16})
    # plot1()
    # plot2()
    # plot3()
    # plot4()
    # plot5()
    # plot6()
    # plot7()
    # plot8()
    # plot9()
    # plot10()
    # plot11()
    # plot12()
    # plot13()
    # plot14()
    # plot15()
    # plot16()
    # plot17()
    # plot18()
    # plot19()
    # plot20()
    plot21_temp()
