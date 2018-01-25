# coding: utf-8

import numpy as np

from keras.layers import Conv2D, Dense, Activation, Flatten, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras import backend as K


def build_encoder():
    model = Sequential()
    model.add(Conv2D(32, (3,3), input_shape=(3, 224, 224)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D())
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3,3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D())
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    return model
