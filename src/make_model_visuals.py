#Author: Gentry Atkinson
#Organization: Texas State University
#Data: 4 February, 2022
#Let's look at some models

#Thanks visualkeras: https://github.com/paulgavrikov/visualkeras

# @misc{Gavrikov2020VisualKeras,
#   author = {Gavrikov, Paul},
#   title = {visualkeras},
#   year = {2020},
#   publisher = {GitHub},
#   journal = {GitHub repository},
#   howpublished = {\url{https://github.com/paulgavrikov/visualkeras}},
# }

import visualkeras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
import tensorflow.keras.metrics as met
from utils.gen_ts_data import generate_pattern_data_as_array
from model_config import loadDic
import numpy as np
from random import randint

def build_cnn(X, num_classes, set, num_channels=1, opt='SGD', loss='mean_squared_error'):
    config_dic = loadDic('CNN')
    print("Input Shape: ", X.shape)
    model = Sequential([
        layers.Input(shape=X[0].shape),
        layers.BatchNormalization(),
        layers.Conv1D(filters=config_dic[set]['l1_numFilters']*1, kernel_size=config_dic[set]['l1_kernelSize'], padding='causal', activation='relu', groups=1),
        layers.Conv1D(filters=config_dic[set]['l1_numFilters']*1, kernel_size=config_dic[set]['l1_kernelSize'], padding='causal', activation='relu', groups=1),
        layers.MaxPooling1D(pool_size=(config_dic[set]['l1_maxPoolSize']*1), data_format='channels_first'),
        layers.Conv1D(filters=config_dic[set]['l2_numFilters'], kernel_size=config_dic[set]['l2_kernelSize'], padding='causal', activation='relu', groups=1),
        layers.Conv1D(filters=config_dic[set]['l2_numFilters'], kernel_size=config_dic[set]['l2_kernelSize'], padding='causal', activation='relu', groups=1),
        layers.MaxPooling1D(pool_size=(config_dic[set]['l2_maxPoolSize']), data_format='channels_first'),
        layers.Dropout(config_dic[set]['dropout']),
        layers.GlobalAveragePooling1D(data_format="channels_first"),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=opt, loss=loss, metrics=[met.CategoricalAccuracy()])
    model.summary()
    return model

if __name__ == '__main__':
    print('### Generate Dummy Data ###')
    X = np.reshape(np.array([generate_pattern_data_as_array(length=150) for _ in range(100)]), (100,1,150))
    y = np.array([randint(0,2) for _ in range(100)])

    print('### Build and Viosualize CNN')
    model = build_cnn(X, 2, 'ss1')
    model.fit(X,y, epochs=1)
    print(type(model))

    print('###Save visualization of CNN###')
    visualkeras.graph_view(model(X[0]), to_file='model_layers/CNN_visualKeras.png')
