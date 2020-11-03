from keras.layers import Input, Lambda, Dense, Flatten
from keras.layers import AveragePooling2D, MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras.models import Model, Sequential
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input

import keras.backend as K
import numpy as np
import matplotlib as plt

from scipy.optimize import fmin_l_bfgs_b

import tensorflow as tf

if tf.__version__.startswith('2'):
    tf.compat.v1.disable_eager_execution()


def VGG16_AvgPool(shape):
    vgg = VGG16(input_shape=shape, weights='imagenet', include_top=False)

    new_model = Sequential()
    for layer in vgg.layers:
        if layer.__class__ == MaxPooling2D:
            new_model.add(AveragePooling2D())
        else:
            new_model.add(layer)

    # i = vgg.input
    # x = i
    # for layer in vgg.layers:
    #     if layer.__class__ == MaxPooling2D:
    #         x = AveragePooling2D()(x)
    #     else:
    #         x = layer(x)

    return new_model

def VGG16_AvgPool_CutOff(shape, num_convs):
    pass