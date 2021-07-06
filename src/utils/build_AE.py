#Author: Gentry Atkinson
#Organization: Texas University
#Data: 25 September, 2020
#Build, compile and return a convolutional autoencoder

#from tensorflow import keras
import numpy as np
import random as rand
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dropout, Dense
from tensorflow.keras.layers import Input, Conv1DTranspose, Lambda, Reshape, BatchNormalization
from tensorflow.keras.layers import UpSampling1D
from tensorflow.keras.callbacks import EarlyStopping
from matplotlib import pyplot as plt

#tf.config.threading.set_inter_op_parallelism_threads(8)

def build_AE(train_example, features_per_channel=30):
    print("Input shape: ", train_example.shape)
    s=[1,1]
    if train_example.ndim==1:
        s[1] = len(train_example)
    else:
        s[0] = train_example.shape[0]
        s[1] = train_example.shape[1]
    model = Sequential([
        Input(shape=train_example.shape),
        BatchNormalization(scale=False),
        Reshape((s[0], s[1])),
        Conv1D(filters=128, kernel_size=16, activation='relu', padding='same'),
        Conv1D(filters=128, kernel_size=16, activation='relu', padding='same'),
        MaxPooling1D(pool_size=(16), data_format='channels_first'),
        Flatten(),
        Dense(features_per_channel*s[0], activation='softmax', name='Embedding'),
        Reshape((features_per_channel,s[0])),
        Conv1DTranspose(filters=128, activation='relu', kernel_size=16, padding='same'),
        Conv1DTranspose(filters=128, activation='relu', kernel_size=16, padding='same'),
        BatchNormalization(),
        Flatten(),
        Dense(s[0] * s[1], activation='linear'),
        Reshape((s[0],s[1]))
    ])
    model.compile(optimizer='RMSprop', loss='mean_squared_error', metrics=[])
    #model.summary()
    print("Output shape: ", model.output.shape)
    return model

def train_AE(ae, train_set):
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
    ae.fit(train_set, train_set, epochs=100, verbose=0, callbacks=[es], validation_split=0.1, batch_size=10)
    return ae

def trim_decoder(autoenc):
    print("Removing decoder from autoencoder....")
    o = autoenc.layers[-8].output
    encoder = Model(autoenc.input, [o])
    encoder.summary()
    return encoder

def get_trained_AE(X, withVisual=False):
    lookAtMe = rand.randint(0, 1000)
    ae = build_AE(X[0])
    ae = train_AE(ae, X)
    if withVisual:
        predict = ae.predict(X)
        plt.plot(range(0,len(X[lookAtMe,0,:])), X[lookAtMe,0,:], c='blue')
        plt.plot(range(0,len(X[lookAtMe,0,0:])), predict[lookAtMe,0,:], c='red')
        plt.title("Plotting Original and Decoded Instance " + str(lookAtMe))
        plt.show()
    ae = trim_decoder(ae)
    return ae

if __name__ == "__main__":
    print("Test model building")
    #rand_input = np.random.ranf((100, 300, 3))
    rand_input = np.genfromtxt('data/synthetic_test_data.csv', delimiter=',')
    #norm = np.linalg.norm(rand_input[0])
    norm = np.max(rand_input)
    print("norm= ", norm)
    rand_input = rand_input/norm
    print ("Size of data: ", (rand_input.shape))
    rand_input = np.resize(rand_input, (len(rand_input), len(rand_input[0]), 1))
    print ("ReSize of data: ", (rand_input.shape))
    ae = build_AE(rand_input[0])
    ae = train_AE(ae, rand_input)
    predict = ae.predict(rand_input)
    print(predict.shape)
    plt.plot(range(0,len(rand_input[3,:,0])), rand_input[3,:,0], c='blue')
    plt.plot(range(0,len(rand_input[3,:,0])), predict[3,:,0], c='red')
    plt.show()
    ae = trim_decoder(ae)
