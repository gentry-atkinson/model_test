#!/usr/bin/env bash

#Author: Gentry Atkinson
#Organization: Texas State University
#Data: 04 September, 2021
#Run every script in the model test

# echo "Processing Datasets"
# python3 src/process_all_data.py > logs/preprocess_log.txt
#
# echo "Making Visualizations"
# python3 src/visualizer.py > logs/visualizer_log.txt

echo "CNN Tests"
python3 -Wignore src/CNN_tests.py > logs/CNN_log.txt

echo "LSTM Tests"
python3 -Wignore src/LSTM_tests.py > logs/LSTM_log.txt

echo "Transformer Tests"
python3 -Wignore src/Transf_tests.py > logs/Transf_log.txt

echo "SVM Tests"
python3 -Wignore src/SVM_tests.py > logs/SVM_log.txt

echo "NB Tests"
python3 -Wignore src/NB_tests.py > logs/NB_log.txt

echo "Tree Tests"
python3 -Wignore src/Tree_tests.py > logs/Tree_log.txt
