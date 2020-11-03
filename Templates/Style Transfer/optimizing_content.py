import warnings

warnings.simplefilter("ignore")

from keras.layers import Input, Lambda, Dense, Flatten
from keras.layers import AveragePooling2D, MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras.models import Model, Sequential
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image

import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt

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
    # there are 13 convolutions in total

    if num_convs < 1 or num_convs > 13:
        print("num_convs must be in the range [1, 13]")
        return None

    model = VGG16_AvgPool(shape)

    new_model = Sequential()

    # n = 0
    # for layer in model.layers:
    #     if layer.__class__ == Conv2D:
    #         n += 1
    #     new_model.add(layer)
    #     if n >= num_convs:
    #         break

    n = 0
    output = None
    for layer in model.layers:
        if layer.__class__ == Conv2D:
            n += 1
        if n >= num_convs:
            output = layer.output
            break

    return Model(model.input, output)


def unpreprocess(img):
    img[..., 0] += 103.939
    img[..., 1] += 116.779
    img[..., 2] += 126.68
    img = img[..., ::-1]
    return img


def scale_img(x):
    x = x - x.min()
    x = x / x.max()
    return x


path = './content/elephant.jpg'
img = image.load_img(path)

# convert image to array and preprocess for vgg
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# (1, 225, 300, 3)
batch_shape = x.shape
shape = x.shape[1:]

plt.imshow(img)
plt.show()

# make a content model
# try different cutoffs to see the images that result
content_model = VGG16_AvgPool_CutOff(shape, 11)

# make the target
target = K.variable(content_model.predict(x))

# try to match the image

# define our loss in keras

loss = K.mean(k.square(target - content_model.output))

# gradients which are needed by the optimizer
