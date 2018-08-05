import numpy as np
import os
import cv2
import tensorflow as tf
import keras.backend as K
from keras.layers import \
    TimeDistributed, Conv2D, ZeroPadding2D, \
    Activation, Input, concatenate, Dropout
from keras.layers.recurrent import LSTM, GRU
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.core import Lambda, Flatten, Dense
import matplotlib.pyplot as plt
from numpy.random import seed
import time as time

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


# Simple Inception - LSTM model
# 2 CNN -> 1 Inception Block -> Dense -> (TimeDistributed) -> 1 LSTM
# -> FC -> Sigmoid
def inception_lstm_model_a(input_shape):
    X_input = Input(input_shape[1:])
    model = inception_network(X_input)
    model = AveragePooling2D(pool_size=(3, 3), strides=(1, 1))(model)
    model = Flatten()(model)
    model = Dense(128, activation="relu", name="dense_1")(model)
    model = Model(inputs=X_input, outputs=model)
    model_input = Input(shape=input_shape)
    model = TimeDistributed(model)(model_input)
    model = LSTM(32)(model)
    model = Dense(1, activation='sigmoid', name='result')(model)
    model = Lambda(lambda x: K.l2_normalize(x,axis=1))(model)

    model = Model(inputs=model_input, outputs=model, name='deep_inception_lstm1')
    return model


inception_lstm_a = inception_lstm_model_a((50, 96, 96, 3))
inception_lstm_a.compile(optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy'])
# model info
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# input_8 (InputLayer)         (None, 400, 96, 96, 3)    0
# _________________________________________________________________
# time_distributed_4 (TimeDist (None, 400, 128)          10413868
# _________________________________________________________________
# lstm_4 (LSTM)                (None, 32)                20608
# _________________________________________________________________
# result (Dense)               (None, 1)                 33
# _________________________________________________________________
# lambda_4 (Lambda)            (None, 1)                 0
# =================================================================
# Total params: 10,434,509
# Trainable params: 10,433,967
# Non-trainable params: 542
# _________________________________________________________________
# None


# More Complicated Inception-LSTM model: 3-layer CNN to 64 LSTM
def inception_lstm_model_b(input_shape):
    X_input = Input(input_shape[1:])
    model = inception_network(X_input)
    model = AveragePooling2D(pool_size=(3, 3), strides=(1, 1))(model)
    model = Flatten()(model)
    model = Dense(128, activation="relu", name="dense_1")(model)
    model = Model(inputs=X_input, outputs=model)
    model_input = Input(shape=input_shape)
    model = TimeDistributed(model)(model_input)
    model = LSTM(64)(model)
    model = Dense(1, activation='sigmoid', name='result')(model)
    model = Lambda(lambda x: K.l2_normalize(x,axis=1))(model)

    model = Model(inputs=model_input, outputs=model, name='deep_inception_lstm2')
    return model


inception_lstm_b = inception_lstm_model_b((50, 96, 96, 3))
inception_lstm_b.compile(optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy'])
# model info
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# input_10 (InputLayer)        (None, 400, 96, 96, 3)    0
# _________________________________________________________________
# time_distributed_5 (TimeDist (None, 400, 128)          10413868
# _________________________________________________________________
# lstm_5 (LSTM)                (None, 64)                49408
# _________________________________________________________________
# result (Dense)               (None, 1)                 65
# _________________________________________________________________
# lambda_5 (Lambda)            (None, 1)                 0
# =================================================================
# Total params: 10,463,341
# Trainable params: 10,462,799
# Non-trainable params: 542
# _________________________________________________________________
# None

# Complicated LSTM model...definitely overkill
def inception_lstm_model_c(input_shape):
    X_input = Input(input_shape[1:])
    model = inception_network(X_input)
    model = AveragePooling2D(pool_size=(3, 3), strides=(1, 1))(model)
    model = Flatten()(model)
    model = Dense(128, activation="relu", name="dense_1")(model)
    model = Model(inputs=X_input, outputs=model)
    model_input = Input(shape=input_shape)
    model = TimeDistributed(model)(model_input)
    model_output = LSTM(42, return_sequences=True)(model)
    model = LSTM(42)(model_output)
    model = Dense(1, activation='sigmoid', name='result')(model)
    model = Lambda(lambda x: K.l2_normalize(x,axis=1))(model)
    model = Model(inputs=model_input, outputs=model, name='deep_inception_lstm3')
    return model


inception_lstm_c = inception_lstm_model_c((50, 96, 96, 3))
inception_lstm_c.compile(optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy'])
# model summary
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# input_12 (InputLayer)        (None, 400, 96, 96, 3)    0
# _________________________________________________________________
# time_distributed_6 (TimeDist (None, 400, 128)          10413868
# _________________________________________________________________
# lstm_6 (LSTM)                (None, 400, 42)           28728
# _________________________________________________________________
# lstm_7 (LSTM)                (None, 42)                14280
# _________________________________________________________________
# result (Dense)               (None, 1)                 43
# _________________________________________________________________
# lambda_6 (Lambda)            (None, 1)                 0
# =================================================================
# Total params: 10,456,919
# Trainable params: 10,456,377
# Non-trainable params: 542
# _________________________________________________________________
# None


# 3-layer CNN to 1 GRU
def inception_gru_model_a(input_shape):
    X_input = Input(input_shape[1:])
    model = inception_network(X_input)
    model = AveragePooling2D(pool_size=(3, 3), strides=(1, 1))(model)
    model = Flatten()(model)
    model = Dense(128, activation="relu", name="dense_1")(model)
    model = Model(inputs=X_input, outputs=model)
    model_input = Input(shape=input_shape)
    model = TimeDistributed(model)(model_input)
    model = GRU(64)(model)
    model = Dense(1, activation='sigmoid', name='result')(model)
    model = Lambda(lambda x: K.l2_normalize(x,axis=1))(model)

    model = Model(inputs=model_input, outputs=model, name='deep_inception_gru')
    return model

inception_gru_a = inception_gru_model_a((50, 96, 96, 3))
inception_gru_a.compile(optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy'])
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# input_14 (InputLayer)        (None, 400, 96, 96, 3)    0
# _________________________________________________________________
# time_distributed_7 (TimeDist (None, 400, 128)          10413868
# _________________________________________________________________
# gru_1 (GRU)                  (None, 64)                37056
# _________________________________________________________________
# result (Dense)               (None, 1)                 65
# _________________________________________________________________
# lambda_7 (Lambda)            (None, 1)                 0
# =================================================================
# Total params: 10,450,989
# Trainable params: 10,450,447
# Non-trainable params: 542
# _________________________________________________________________
# None

# 3 layer CNN only--note this is only compatible with individual images,
# not video!! based off of:
# http://www.ee.cityu.edu.hk/~lmpo/publications/2018_ESA_LiveNet.pdf
def simple_model(input_shape):
    X_input = Input(input_shape)
    model = starter_layers(X_input)
    model = Conv2D(128, (3,3), name='simp_4conv')(model)
    model = BatchNormalization(axis=1, epsilon=0.00001, name='simp_4bn')(model)
    model = Activation('relu')(model)
    model = AveragePooling2D(pool_size = (3,3), strides=(1,1))(model)
    model = Flatten()(model)
    model = Dense(16, activation='relu')(model)
    model = Dropout(0.5)(model)
    model = Dense(9, activation='relu')(model)
    model = Dropout(0.25)(model)
    model = Dense(1, activation='sigmoid')(model)

    model = Model(inputs=X_input, outputs=model, name="simple CNN model")
    return model


simple = simple_model((96, 96, 3))
simple.compile(optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy'])
# model summary
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# input_18 (InputLayer)        (None, 96, 96, 3)         0
# _________________________________________________________________
# zero_padding2d_97 (ZeroPaddi (None, 102, 102, 3)       0
# _________________________________________________________________
# conv1 (Conv2D)               (None, 48, 48, 64)        9472
# _________________________________________________________________
# bn1 (BatchNormalization)     (None, 48, 48, 64)        192
# _________________________________________________________________
# activation_146 (Activation)  (None, 48, 48, 64)        0
# _________________________________________________________________
# zero_padding2d_98 (ZeroPaddi (None, 50, 50, 64)        0
# _________________________________________________________________
# max_pooling2d_35 (MaxPooling (None, 24, 24, 64)        0
# _________________________________________________________________
# conv2 (Conv2D)               (None, 24, 24, 64)        4160
# _________________________________________________________________
# bn2 (BatchNormalization)     (None, 24, 24, 64)        96
# _________________________________________________________________
# activation_147 (Activation)  (None, 24, 24, 64)        0
# _________________________________________________________________
# zero_padding2d_99 (ZeroPaddi (None, 26, 26, 64)        0
# _________________________________________________________________
# conv3 (Conv2D)               (None, 24, 24, 192)       110784
# _________________________________________________________________
# bn3 (BatchNormalization)     (None, 24, 24, 192)       96
# _________________________________________________________________
# activation_148 (Activation)  (None, 24, 24, 192)       0
# _________________________________________________________________
# zero_padding2d_100 (ZeroPadd (None, 26, 26, 192)       0
# _________________________________________________________________
# max_pooling2d_36 (MaxPooling (None, 12, 12, 192)       0
# _________________________________________________________________
# simp_4conv (Conv2D)          (None, 10, 10, 128)       221312
# _________________________________________________________________
# simp_4bn (BatchNormalization (None, 10, 10, 128)       40
# _________________________________________________________________
# activation_149 (Activation)  (None, 10, 10, 128)       0
# _________________________________________________________________
# average_pooling2d_17 (Averag (None, 8, 8, 128)         0
# _________________________________________________________________
# flatten_10 (Flatten)         (None, 8192)              0
# _________________________________________________________________
# dense_5 (Dense)              (None, 16)                131088
# _________________________________________________________________
# dropout_3 (Dropout)          (None, 16)                0
# _________________________________________________________________
# dense_6 (Dense)              (None, 9)                 153
# _________________________________________________________________
# dropout_4 (Dropout)          (None, 9)                 0
# _________________________________________________________________
# dense_7 (Dense)              (None, 1)                 10
# =================================================================
# Total params: 477,403
# Trainable params: 477,191
# Non-trainable params: 212
# _________________________________________________________________
# None

# 3 layer Simple CNN to LSTM

def simple_LSTM_model(input_shape):
    X_input = Input(input_shape[1:])
    model = starter_layers(X_input)
    model = Conv2D(128, (3,3), name='simp_4conv')(model)
    model = BatchNormalization(axis=1, epsilon=0.00001, name='simp_4bn')(model)
    model = Activation('relu')(model)
    model = AveragePooling2D(pool_size = (3,3), strides=(1,1))(model)
    model = Flatten()(model)
    model = Dense(256, activation='relu')(model)
    model = Dropout(0.5)(model)
    model = Model(inputs=X_input, outputs=model)
    model_input = Input(input_shape)
    model = TimeDistributed(model)(model_input)
    model = LSTM(128)(model)
    model = Dense(1, activation='sigmoid')(model)

    model = Model(inputs=model_input, outputs=model, name="simple CNN LSTM")
    return model


simple_LSTM = simple_LSTM_model((50, 96, 96, 3))
simple_LSTM.compile(optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy'])

# model info
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# input_22 (InputLayer)        (None, 400, 96, 96, 3)    0
# _________________________________________________________________
# time_distributed_9 (TimeDist (None, 400, 256)          2443560
# _________________________________________________________________
# lstm_9 (LSTM)                (None, 128)               197120
# _________________________________________________________________
# dense_11 (Dense)             (None, 1)                 129
# =================================================================
# Total params: 2,640,809
# Trainable params: 2,640,597
# Non-trainable params: 212
# _________________________________________________________________
# None

def simple_LSTM_model_2(input_shape):
    X_input = Input(input_shape[1:])
    model = starter_layers(X_input)
    model = Conv2D(128, (3,3), name='simp_4conv')(model)
    model = BatchNormalization(axis=1, epsilon=0.00001, name='simp_4bn')(model)
    model = Activation('relu')(model)
    model = AveragePooling2D(pool_size = (3,3), strides=(1,1))(model)
    model = Flatten()(model)
    model = Dense(256, activation='relu')(model)
    model = Dropout(0.5)(model)
    model = Model(inputs=X_input, outputs=model)
    model_input = Input(input_shape)
    model = TimeDistributed(model)(model_input)
    model = LSTM(64)(model)
    model = Dense(1, activation='sigmoid')(model)

    model = Model(inputs=model_input, outputs=model, name="simple CNN LSTM")
    return model


simple_LSTM_2 = simple_LSTM_model_2((50, 96, 96, 3))
simple_LSTM_2.compile(optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy'])
# model summary
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# input_28 (InputLayer)        (None, 50, 96, 96, 3)     0
# _________________________________________________________________
# time_distributed_13 (TimeDis (None, 50, 256)           2443560
# _________________________________________________________________
# lstm_11 (LSTM)               (None, 64)                82176
# _________________________________________________________________
# dense_13 (Dense)             (None, 1)                 65
# =================================================================
# Total params: 2,525,801
# Trainable params: 2,525,589
# Non-trainable params: 212
# _________________________________________________________________
# None

def simple_GRU_model(input_shape):
    X_input = Input(input_shape[1:])
    model = starter_layers(X_input)
    model = Conv2D(128, (3,3), name='simp_4conv')(model)
    model = BatchNormalization(axis=1, epsilon=0.00001, name='simp_4bn')(model)
    model = Activation('relu')(model)
    model = AveragePooling2D(pool_size = (3,3), strides=(1,1))(model)
    model = Flatten()(model)
    model = Dense(256, activation='relu')(model)
    model = Dropout(0.5)(model)
    model = Model(inputs=X_input, outputs=model)
    model_input = Input(input_shape)
    model = TimeDistributed(model)(model_input)
    model = GRU(128)(model)
    model = Dense(1, activation='sigmoid')(model)

    model = Model(inputs=model_input, outputs=model, name="simple CNN LSTM")
    return model


simple_GRU = simple_GRU_model((50, 96, 96, 3))
simple_GRU.compile(optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy'])
# model summary
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# input_24 (InputLayer)        (None, 400, 96, 96, 3)    0
# _________________________________________________________________
# time_distributed_10 (TimeDis (None, 400, 256)          2443560
# _________________________________________________________________
# gru_2 (GRU)                  (None, 128)               147840
# _________________________________________________________________
# dense_13 (Dense)             (None, 1)                 129
# =================================================================
# Total params: 2,591,529
# Trainable params: 2,591,317
# Non-trainable params: 212
# _________________________________________________________________
# None

def simple_GRU_model_2(input_shape):
    X_input = Input(input_shape[1:])
    model = starter_layers(X_input)
    model = Conv2D(128, (3,3), name='simp_4conv')(model)
    model = BatchNormalization(axis=1, epsilon=0.00001, name='simp_4bn')(model)
    model = Activation('relu')(model)
    model = AveragePooling2D(pool_size = (3,3), strides=(1,1))(model)
    model = Flatten()(model)
    model = Dense(256, activation='relu')(model)
    model = Dropout(0.5)(model)
    model = Model(inputs=X_input, outputs=model)
    model_input = Input(input_shape)
    model = TimeDistributed(model)(model_input)
    model = GRU(64)(model)
    model = Dense(1, activation='sigmoid')(model)

    model = Model(inputs=model_input, outputs=model, name="simple CNN LSTM")
    return model

simple_GRU_2 = simple_GRU_model_2((50, 96, 96, 3))
simple_GRU_2.compile(optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy'])
# model summary
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# input_30 (InputLayer)        (None, 50, 96, 96, 3)     0
# _________________________________________________________________
# time_distributed_14 (TimeDis (None, 50, 256)           2443560
# _________________________________________________________________
# gru_5 (GRU)                  (None, 64)                61632
# _________________________________________________________________
# dense_15 (Dense)             (None, 1)                 65
# =================================================================
# Total params: 2,505,257
# Trainable params: 2,505,045
# Non-trainable params: 212
# _________________________________________________________________
# None


def check_model(model, X_input):
    start = time.time()
    print model.predict(x=X_input)
    return time.time() - start


r = seed(1)
X_input = np.random.rand(1, 50, 96, 96, 3)
check_model(simple_GRU_2, X_input)
# times for models to run on one forward pass on my mac (2.8 GHz)
# simple CNN: 0.013265132904052734
# simple GRU: 0.7893438339233398
# simple GRU-2: 0.805243968963623
# simple LSTM 2: 0.8088140487670898
# simple LSTM: 0.8190209865570068
# Inception-LSTM c: 1.3269569873809814
# Inception-GRU a: 1.3289849758148193
# Inception-LSTM b: 1.337082862854004
# Inception-LSTM a: 1.3395531177520752
