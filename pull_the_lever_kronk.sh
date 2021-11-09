#!/usr/bin/env bash

#Author: Gentry Atkinson
#Organization: Texas University
#Data: 04 September, 2021
#Run every script in the model test

# echo "Processing Datasets"
# python3 src/process_all_data.py > logs/preprocess_log.txt
#
# echo "Making Visualizations"
# python3 src/visualizer.py > logs/visualizer_log.txt

echo "CNN Tests"
python3 src/CNN_tests.py > logs/CNN_log.txt

echo "LSTM Tests"
python3 src/LSTM_tests.py > logs/LSTM_log.txt

echo "SVM Tests"
python3 src/SVM_tests.py > logs/SVM_log.txt

echo "NB Tests"
python3 src/NB_tests.py > logs/NB_log.txt

echo "Tree Tests"
python3 src/Tree_tests.py > logs/Tree_log.txt
