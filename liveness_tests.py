import tensorflow as tf
import numpy as np
import os
import cv2
from numpy import genfromtxt
import keras.backend as K 
from keras.layers import TimeDistributed, Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.layers.recurrent import LSTM, GRU
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.core import Lambda, Flatten, Dense
import h5py
import matplotlib.pyplot as plt

def conv2d_bn(x,
              layer=None,
              cv1_out=None,
              cv1_filter=(1, 1),
              cv1_strides=(1, 1),
              cv2_out=None,
              cv2_filter=(3, 3),
              cv2_strides=(1, 1),
              padding=None):
    num = '' if cv2_out == None else '1'
    tensor = Conv2D(cv1_out, cv1_filter, strides=cv1_strides, name=layer+'_conv'+num)(x)
    tensor = BatchNormalization(axis=1, epsilon=0.00001, name=layer+'_bn'+num)(tensor)
    tensor = Activation('relu')(tensor)
    if padding == None:
        return tensor
    tensor = ZeroPadding2D(padding=padding)(tensor)
    if cv2_out == None:
        return tensor
    tensor = Conv2D(cv2_out, cv2_filter, strides=cv2_strides, name=layer+'_conv'+'2')(tensor)
    tensor = BatchNormalization(axis=1, epsilon=0.00001, name=layer+'_bn'+'2')(tensor)
    tensor = Activation('relu')(tensor)
    return tensor

def starter_layers(X_input):
    # Zero-Padding
    X = ZeroPadding2D((3, 3))(X_input)

    # First Block
    X = Conv2D(64, (7, 7), strides = (2, 2), name = 'conv1')(X)
    X = BatchNormalization(axis = 1, name = 'bn1')(X)
    X = Activation('relu')(X)

    # Zero-Padding + MAXPOOL
    X = ZeroPadding2D((1, 1))(X)
    X = MaxPooling2D((3, 3), strides = 2)(X)

    # Second Block
    X = Conv2D(64, (1, 1), strides = (1, 1), name = 'conv2')(X)
    X = BatchNormalization(axis = 1, epsilon=0.00001, name = 'bn2')(X)
    X = Activation('relu')(X)

    # Zero-Padding + MAXPOOL
    X = ZeroPadding2D((1, 1))(X)

    # Second Block
    X = Conv2D(192, (3, 3), strides = (1, 1), name = 'conv3')(X)
    X = BatchNormalization(axis = 1, epsilon=0.00001, name = 'bn3')(X)
    X = Activation('relu')(X)

    # Zero-Padding + MAXPOOL
    X = ZeroPadding2D((1, 1))(X)
    X = MaxPooling2D(pool_size = 3, strides = 2)(X)

    return X

def inception_block_3(X):


    X_3x3 = Conv2D(96, (1, 1), name ='inception_3a_3x3_conv1')(X)
    X_3x3 = BatchNormalization(axis=1, epsilon=0.00001, name = 'inception_3a_3x3_bn1')(X_3x3)
    X_3x3 = Activation('relu')(X_3x3)
    X_3x3 = ZeroPadding2D(padding=(1, 1))(X_3x3)
    X_3x3 = Conv2D(128, (3, 3), name='inception_3a_3x3_conv2')(X_3x3)
    X_3x3 = BatchNormalization(axis=1, epsilon=0.00001, name='inception_3a_3x3_bn2')(X_3x3)
    X_3x3 = Activation('relu')(X_3x3)

    X_5x5 = Conv2D(16, (1, 1), name='inception_3a_5x5_conv1')(X)
    X_5x5 = BatchNormalization(axis=1, epsilon=0.00001, name='inception_3a_5x5_bn1')(X_5x5)
    X_5x5 = Activation('relu')(X_5x5)
    X_5x5 = ZeroPadding2D(padding=(2, 2))(X_5x5)
    X_5x5 = Conv2D(32, (5, 5), name='inception_3a_5x5_conv2')(X_5x5)
    X_5x5 = BatchNormalization(axis=1, epsilon=0.00001, name='inception_3a_5x5_bn2')(X_5x5)
    X_5x5 = Activation('relu')(X_5x5)

    X_pool = MaxPooling2D(pool_size=3, strides=2)(X)
    X_pool = Conv2D(32, (1, 1), name='inception_3a_pool_conv')(X_pool)
    X_pool = BatchNormalization(axis=1, epsilon=0.00001, name='inception_3a_pool_bn')(X_pool)
    X_pool = Activation('relu')(X_pool)
    X_pool = ZeroPadding2D(padding=((3, 4), (3, 4)))(X_pool)

    X_1x1 = Conv2D(64, (1, 1), name='inception_3a_1x1_conv')(X)
    X_1x1 = BatchNormalization(axis=1, epsilon=0.00001, name='inception_3a_1x1_bn')(X_1x1)
    X_1x1 = Activation('relu')(X_1x1)

    # CONCAT
#     arr = [X_3x3, X_5x5, X_pool, X_1x1]
#     for layer in arr:
#         print layer.shape
    inception = concatenate([X_3x3, X_5x5, X_pool, X_1x1], axis=-1)

    X_3x3 = Conv2D(96, (1, 1), name='inception_3b_3x3_conv1')(inception)
    X_3x3 = BatchNormalization(axis=1, epsilon=0.00001, name='inception_3b_3x3_bn1')(X_3x3)
    X_3x3 = Activation('relu')(X_3x3)
    X_3x3 = ZeroPadding2D(padding=(1, 1))(X_3x3)
    X_3x3 = Conv2D(128, (3, 3), name='inception_3b_3x3_conv2')(X_3x3)
    X_3x3 = BatchNormalization(axis=1, epsilon=0.00001, name='inception_3b_3x3_bn2')(X_3x3)
    X_3x3 = Activation('relu')(X_3x3)

    X_5x5 = Conv2D(32, (1, 1), name='inception_3b_5x5_conv1')(inception)
    X_5x5 = BatchNormalization(axis=1, epsilon=0.00001, name='inception_3b_5x5_bn1')(X_5x5)
    X_5x5 = Activation('relu')(X_5x5)
    X_5x5 = ZeroPadding2D(padding=(2, 2))(X_5x5)
    X_5x5 = Conv2D(64, (5, 5), name='inception_3b_5x5_conv2')(X_5x5)
    X_5x5 = BatchNormalization(axis=1, epsilon=0.00001, name='inception_3b_5x5_bn2')(X_5x5)
    X_5x5 = Activation('relu')(X_5x5)

    X_pool = AveragePooling2D(pool_size=(3, 3), strides=(3, 3))(inception)
    X_pool = Conv2D(64, (1, 1), name='inception_3b_pool_conv')(X_pool)
    X_pool = BatchNormalization(axis=1, epsilon=0.00001, name='inception_3b_pool_bn')(X_pool)
    X_pool = Activation('relu')(X_pool)
    X_pool = ZeroPadding2D(padding=(4, 4))(X_pool)

    X_1x1 = Conv2D(64, (1, 1), name='inception_3b_1x1_conv')(inception)
    X_1x1 = BatchNormalization(axis=1, epsilon=0.00001, name='inception_3b_1x1_bn')(X_1x1)
    X_1x1 = Activation('relu')(X_1x1)

    inception = concatenate([X_3x3, X_5x5, X_pool, X_1x1], axis=-1)
    print inception.shape
    
    X_3x3 = conv2d_bn(inception,
        layer='inception_5b_3x3',
        cv1_out=96,
        cv1_filter=(1, 1),
        cv2_out=384,
        cv2_filter=(3, 3),
        cv2_strides=(1, 1),
        padding=(1, 1))
    
    
    X_pool = MaxPooling2D(pool_size=3, strides=1)(inception)
    X_pool = conv2d_bn(X_pool,
                           layer='inception_5b_pool',
                           cv1_out=96,
                           cv1_filter=(1, 1))
    X_pool = ZeroPadding2D(padding=(1, 1))(X_pool)

    X_1x1 = conv2d_bn(inception,
                           layer='inception_5b_1x1',
                           cv1_out=256,
                           cv1_filter=(1, 1))
    inception = concatenate([X_3x3, X_pool, X_1x1], axis=-1)

    return inception

def inception_network(X_input):
    X = starter_layers(X_input)
    X = inception_block_3(X)
    return X

def inception_lstm_model_a(input_shape):
    print "START"
    X_input = Input(input_shape[1:])
    model = inception_network(X_input)
    model = AveragePooling2D(pool_size = (3,3), strides=(1,1))(model)
    model = Flatten()(model)
    model = Dense(128, activation="relu", name="dense_1")(model)
    model = Model(inputs=X_input, outputs=model)
    model_input = Input(shape=input_shape)
    model = TimeDistributed(model)(model_input)
    model = LSTM(32)(model)
    model = Dense(9, activation="relu", name="dense_2",
        activity_regularizer=l2(0.01))(model)
    model = Dense(1, activation='sigmoid', name='result')(model)
    model = Lambda(lambda x: K.l2_normalize(x,axis=1))(model)

    model = Model(inputs=model_input, outputs=model, name='deep_inception_lstm')
    return model

m = inception_lstm_model_a((400, 96, 96, 3))
m.compile(optimizer='adam', loss='binary_crossentropy', metrics = ['accuracy'])
# 3-layer CNN to 2 LSTM

# 3-layer CNN to 1 GRU

# 3-layer CNN to 2 GRU


# 4 layer only
