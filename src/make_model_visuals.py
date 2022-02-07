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
from tensorflow.python.keras.models import Sequential, Model
from  tensorflow.python.keras.engine.functional import ModuleWrapper
from tensorflow.keras.layers import BatchNormalization, LayerNormalization
from tensorflow.python.keras import layers
import tensorflow.python.keras.metrics as met
from utils.gen_ts_data import generate_pattern_data_as_array
from model_config import loadDic
import numpy as np
from random import randint
# from tensorflow.keras.utils import to_categorical
from PIL import ImageFont
from collections import defaultdict
import matplotlib

def build_cnn(X, num_classes, set, num_channels=1, opt='SGD', loss='mean_squared_error'):
    config_dic = loadDic('CNN')
    print("Input Shape: ", X.shape)
    model = Sequential([
        layers.Input(shape=X[0].shape),
        BatchNormalization(),
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

def build_lstm(X, num_classes, set, opt='SGD', loss='mean_squared_error'):
    config_dic = loadDic('LSTM')
    print("Input Shape: ", X.shape)
    model = Sequential([
        layers.Input(shape=X[0].shape),
        BatchNormalization(),
        layers.LSTM(config_dic[set]['lstm_units'], recurrent_dropout=config_dic[set]['dropout'], activation='relu'),
        layers.Dropout(config_dic[set]['dropout']),
        layers.Dense(config_dic[set]['hidden_dense_size'], activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=opt, loss=loss, metrics=[met.CategoricalAccuracy()])
    model.summary()
    return model

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    """
    TransformerEncoder for time-series encoder, conforming to the
    transformer architecture from Attention Is All You Need (Vaswani 2017)
    https://arxiv.org/abs/1706.03762
    @param (int) num_heads: Number of Attention Heads
    @param (int) head_size : Head Size
    @param (int) ff_dim: Feed Forward Dimension
    @param (float) dropout : Dropout (between 0 and .99)
    
    Return: transformer encoder
    """

    # Normalization and Attention
    print(type(inputs))
    x = BatchNormalization()(inputs)
    x = layers.MultiHeadAttention(num_heads=num_heads, key_dim=head_size, dropout=dropout)(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation='relu')(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res 

def build_tran(X, num_classes, set, opt='SGD', loss='mean_squared_error'):
    print("Input Shape: ", X.shape)
    config_dic = loadDic('Transformer')
    inputs = layers.Input(shape=X[0].shape)
    x = inputs
    for _ in range(config_dic[set]['num_attn+layers']):
        x = transformer_encoder(x, config_dic[set]['head_size'], config_dic[set]['num_heads'], config_dic[set]['ff_dim'], config_dic[set]['dropout'])
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Flatten()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = Model(inputs, outputs)
    model.compile(optimizer=opt, loss=loss, metrics=[met.CategoricalAccuracy()])
    model.summary()
    return model

if __name__ == '__main__':
    print('### Generate Dummy Data ###')
    X = np.reshape(np.array([generate_pattern_data_as_array(length=150) for _ in range(100)]), (100,1,150))
    y = np.array([randint(0,1) for _ in range(100)])
    config_dic = loadDic('CNN')
    print(X.shape)

    print('### Build and Viosualize CNN')
    model = build_cnn(X, 2, 'ss1')

    model.summary()

    color_map = defaultdict(dict)
    color_map[layers.Conv1D]['fill'] = 'orange'
    color_map[BatchNormalization]['fill'] = 'gray'
    color_map[layers.Dropout]['fill'] = 'pink'
    color_map[layers.MaxPooling1D]['fill'] = 'red'
    color_map[layers.Dense]['fill'] = 'green'
    color_map[layers.Flatten]['fill'] = 'teal'
    color_map[layers.LSTM]['fill'] = 'blue'
    color_map[layers.GlobalAveragePooling1D]['fill'] = 'purple'
    color_map[layers.MultiHeadAttention]['fill'] = 'yellow'

    ignore = [BatchNormalization, layers.Input, layers.InputLayer, layers.core.TFOpLambda, ModuleWrapper]

    system_fonts = matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
    print(system_fonts)
    font = ImageFont.truetype('LiberationSerif-Regular.ttf', size = 20)   

    print('###Save visualization of CNN###')
    visualkeras.layered_view(model, to_file='imgs/model_layers/CNN_visualkeras.png',  legend=True, scale_xy=1, scale_z=50, color_map=color_map, type_ignore=ignore, font=font)
    visualkeras.graph_view(model, to_file='imgs/model_layers/CNN_graph_visualkeras.png')

    print('### Build and Viosualize LSTM')
    model = build_lstm(X, 2, 'ss1')
    print(model.layers)
    print('###Save visualization of LSTM###')
    visualkeras.layered_view(model, to_file='imgs/model_layers/LSTM_visualkeras.png',  legend=True, scale_xy=.5, scale_z=2, color_map=color_map, type_ignore=ignore, font=font)
    visualkeras.graph_view(model, to_file='imgs/model_layers/LSTM_graph_visualkeras.png')

    print('### Build and Viosualize Transformer')
    model = build_tran(X, 2, 'ss1')

    print('###Save visualization of Transformer###')
    visualkeras.layered_view(model, to_file='imgs/model_layers/Transformer_visualkeras.png',  legend=True, scale_xy=.5, scale_z=2, color_map=color_map, type_ignore=ignore, font=font)
    # visualkeras.graph_view(model, to_file='imgs/model_layers/Transformer_graph_visualkeras.png')