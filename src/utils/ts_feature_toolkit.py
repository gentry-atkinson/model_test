#Author: Gentry Atkinson
#Organization: Texas University
#Data: 22 May, 2020
#This a small library of functions for extracting features from time series datagen

import scipy
from scipy.fft import fft
from scipy import signal
import numpy as np
from tsfresh.feature_extraction import feature_calculators as fc
from tsfresh.utilities.dataframe_functions import impute
from sklearn.metrics import confusion_matrix
from joblib import Parallel, delayed
from functools import reduce
import os

"""
Calculate Apparent Error Rate
Parameters:
    y_true: assigned labels
    y_pred: predicted labels
Returns: the probabilty that a predicted label does not equal the assigned label
"""
def calc_AER(y_true, y_pred):
    assert y_true.ndim == 1, "AER received labels with {} dimensions".format(y_true.ndim)
    assert y_pred.ndim == 1, "AER received labels with {} dimensions".format(y_pred.ndim)
    assert len(y_true) == len(y_pred), "y_true and y_pred must have the same length"
    wrong = np.sum(y_true != y_pred)
    total = len(y_true)
    return wrong/total

"""
Calculate True Error Rate
Parameters:
    aer: probability that assigend label does not equal predicted label
    mlr: probability that assigned label does not equal true class
Returns: the probabilty that a predicted label does not equal the true class
    exact, low, high
Note: the exact value can be used when mlr and aer are independent
"""
def calc_TER(aer, mlr):
    assert aer<=1 and mlr<=1, "Why would an error rate be greater than 1???"
    assert mlr != 0.5, "Sorry, MLR can't be one half"
    #return '(' + str((aer-mlr)/(1-2*mlr)) + ', ' + str(aer-mlr) + ', ' +  str(aer+mlr) + ')'
    return ('{:.3f}, {:.3f}, {:.3f}'.format((aer-mlr)/(1-2*mlr), aer-mlr, aer+mlr))

"""
Calculate Error Rates
Parameters:
    y_true: assigned labels
    y_pred: predicted labels
Returns: the false positive rate and false negative rate
"""
def calc_error_rates(y_true, y_pred):
  conf_matrix = confusion_matrix(y_true, y_pred)
  FP = conf_matrix.sum(axis=0) - np.diag(conf_matrix)
  FN = conf_matrix.sum(axis=1) - np.diag(conf_matrix)
  TP = np.diag(conf_matrix)
  TN = conf_matrix.sum() - (FP + FN + TP)
  FPR = FP / (FP  + TN)
  FNR = FN / (FN + TP)
  return FPR, FNR

def my_special_variance(error_rates):
    assert error_rates.ndim == 2, 'Error rates must be 2 dimensional'
    my_special_mean = error_rates.sum(axis=0)/[len(error_rates), len(error_rates)]
    sum_of_squares = 0
    for e in error_rates:
        sum_of_squares += ((my_special_mean[0]-e[0])**2 + (my_special_mean[1]-e[1])**2)
    return sum_of_squares/len(error_rates)

def dis_from_sym(p):
    return np.mean(np.abs(p[:, 0] - p[:, 1]))

"""
Calculate Bias Metrics
Parameters:
    base_fpr: false positive rate of base model
    base_fnr: false negative rate of base model
    fpr: false positive rate of comparison model
    fnr: false negative rate of comparison model
Returns: CEV and SDE bias measures
Note: see Measure Twice Cut Once https://arxiv.org/pdf/2110.04397.pdf for
    an explanation of CEV and SDE
"""
def calc_bias_metrics(base_fpr, base_fnr, fpr, fnr):
    # get method stats
    FPR_change = (fpr - base_fpr)/base_fpr
    FNR_change = (fnr - base_fnr)/base_fnr
    # make class points
    points = np.dstack((FPR_change, FNR_change))[0]
    #we should use my hand-coded variance, or mse
    return my_special_variance(points), dis_from_sym(points)

def get_normalized_signal_energy(X):
    return np.mean(np.square(X))

def get_zero_crossing_rate(X):
    mean = np.mean(X)
    zero_mean_signal = np.subtract(X, mean)
    return np.mean(np.absolute(np.edif1d(np.sign(X))))

def clean_nan_and_inf(X):
    assert X.ndim == 2, "Please only de-nan 2D sets"
    print('Cleaning nan and inf in array with shape: ', X.shape)
    max_val = np.nanmax(X, axis=None)
    NUM_CORES = os.cpu_count()
    def fix_num(a):
        for col in range(len(X[0])):
            if np.isnan(a):
                return  0
            elif np.isposinf(a):
                return max_val
            elif np.isposinf(a):
               return -1*max_val
            else:
                return a
    # for row in range(len(X)):
    #     for col in range(len(X[0])):
    #         if np.isnan(X[row][col]):
    #             X[row][col] = 0
    #         elif np.isposinf(X[row][col]):
    #             X[row][col] = max_val
    #         elif np.isposinf(X[row][col]):
    #             X[row][col] = -1*max_val
    new_X = Parallel(n_jobs=NUM_CORES)(delayed(fix_num)(i) for i in np.nditer(X, flags=['multi_index']))
    new_X = np.reshape(new_X, X.shape)
    print(np.sum(X != new_X, axis=None), " nans or infs cleaned")
    print("First instance of nanny data: ", X[0])
    print("First instance of nanless data: ", new_X[0])
    return new_X


def get_features_from_one_signal(X, sample_rate=50):
    assert X.ndim ==1, "Expected single signal in feature extraction"
    mean = np.mean(X)
    stdev = np.std(X)
    abs_energy = fc.abs_energy(X)
    sum_of_changes = fc.absolute_sum_of_changes(X)
    autoc = fc.autocorrelation(X, sample_rate)
    count_above_mean = fc.count_above_mean(X)
    count_below_mean = fc.count_below_mean(X)
    kurtosis = fc.kurtosis(X)
    longest_above = fc.longest_strike_above_mean(X)
    zero_crossing = fc.number_crossing_m(X, mean)
    num_peaks = fc.number_peaks(X, int(sample_rate/10))
    sample_entropy = fc.sample_entropy(X)
    spectral_density = fc.spkt_welch_density(X, [{"coeff":1}, {"coeff":2}, {"coeff":3}, {"coeff":4}, {"coeff":5}, {"coeff":6}])
    c, v = zip(*spectral_density)
    v = np.asarray(v)

    return [
        0 if np.isnan(mean) else mean,
        0 if np.isnan(stdev) else stdev,
        0 if np.isnan(abs_energy) else abs_energy,
        0 if np.isnan(sum_of_changes) else sum_of_changes,
        0 if np.isnan(autoc) else autoc,
        0 if np.isnan(count_above_mean) else count_above_mean,
        0 if np.isnan(count_below_mean) else count_below_mean,
        0 if np.isnan(kurtosis) else kurtosis,
        0 if np.isnan(longest_above) else longest_above,
        0 if np.isnan(zero_crossing) else zero_crossing,
        0 if np.isnan(num_peaks) else num_peaks,
        0 if np.isnan(sample_entropy) else sample_entropy,
        v[0], v[1], v[2], v[3], v[4], v[5]
    ]

"""
Get Features for Set
Parameters:
    X: a raw attribut set
    sample_rate: the collection rate of sensor data (currently unused)
    num_instances: the number of instances in the set
Returns: the extracted feature set as a 2D array
"""
def get_features_for_set(X, sample_rate=50, num_instances=None):
    sample_length = len(X[0])
    if num_instances is None:
        num_instances = len(X)
    fet = np.zeros((num_instances, 18))
    NUM_CORES = os.cpu_count()
    fet = Parallel(n_jobs=NUM_CORES)(delayed(get_features_from_one_signal)(i) for i in X)
    return np.array(fet)

#just a little testing section down here
if __name__ == "__main__":
    y_true = np.array([0,1,0,1,0,1,0,1,0,1])
    y_pred = np.array([0,1,1,1,0,1,0,1,0,1])
    aer = calc_AER(y_true, y_pred)
    mlr = 0.05
    ter = calc_TER(aer, mlr)
    print("My apparent error rate: ", aer)
    print("My mislabel rate is: ", mlr)
    print("My true error rate is: ", ter)
