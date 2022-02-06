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
    config_dic = loadDic('CNN')
    print(X.shape)

    print('### Build and Viosualize CNN')
    model = Sequential()
    model.add(layers.InputLayer(input_shape=X[0].shape))
    model.add(layers.Reshape((1,150)))
    model.add(layers.Conv1D(filters=64, activation='relu', padding='valid', kernel_size=(3), data_format='channels_first'))
    model.add(layers.Conv1D(filters=64, activation='relu', padding='valid', kernel_size=(3), data_format='channels_first'))
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    visualkeras.layered_view(model, to_file='imgs/model_layers/test.png', min_xy=10, min_z=10, scale_xy=100, scale_z=100, one_dim_orientation='x')

    print('###Save visualization of CNN###')
    #visualkeras.layered_view(model, to_file='imgs/model_layers/test.png')
