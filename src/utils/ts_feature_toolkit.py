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
from joblib import Parallel, delayed
from functools import reduce
import os

def calc_AER(y_true, y_pred):
    assert y_true.ndim == 1, "AER received labels with {} dimension".format(y_true.ndim)
    assert len(y_true) == len(y_pred), "y_true and y_pred must have the same length"
    wrong = np.sum(y_true != y_pred)
    total = len(y_true)
    return wrong/total

def calc_TER(aer, mlr):
    assert aer<=1 and mlr<=1, "Why would an error rate be greater than 1???"
    assert mlr != 0.5, "Sorry, MLR can't be one half"
    return (aer-mlr)/(1-2*mlr)

def get_normalized_signal_energy(X):
    return np.mean(np.square(X))

def get_zero_crossing_rate(X):
    mean = np.mean(X)
    zero_mean_signal = np.subtract(X, mean)
    return np.mean(np.absolute(np.edif1d(np.sign(X))))

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
        count_above_mean,
        count_below_mean,
        kurtosis,
        longest_above,
        zero_crossing,
        num_peaks,
        sample_entropy,
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

if __name__ == "__main__":
    y_true = np.array([0,1,0,1,0,1,0,1,0,1])
    y_pred = np.array([0,1,1,1,0,1,0,1,0,1])
    aer = calc_AER(y_true, y_pred)
    mlr = 0.05
    ter = calc_TER(aer, mlr)
    print("My apparent error rate: ", aer)
    print("My mislabel rate is: ", mlr)
    print("My true error rate is: ", ter)
