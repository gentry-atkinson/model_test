#Author: Gentry Atkinson
#Organization: Texas University
#Data: 12 July, 2021
#Read the 6 datasets, write in nice format, and then write noisy label settings
#This is going to be a big one_hot_encode

#6 Datasets: 2 synthetic, 2 HAR, 2 BioSignal
#Each dataset will have 7 label sets:
#   -clean
#   -low noise NCAR
#   -high noise NCAR
#   -low noise NAR
#   -high noise NAR
#   -low noise NNAR
#   -high noise NNAR

from utils.gen_ts_data import generate_pattern_data_as_dataframe
import numpy as np

#Create Synthetic Set 1
#Create label sets for SS1

#Create Synthetic Set 2
#Create label sets for SS2

#Use Lee's files to get HAR Set 1
#Create label sets for HAR1

#Process UCI HAR inertial signals into a good file
#Create label sets for HAR2

#Process Sleep Apnea set into BioSignal Set 1
#Create label sets for BS1

#Process PD Gait set into BioSignal Set 2
#Create label sets for BS2
