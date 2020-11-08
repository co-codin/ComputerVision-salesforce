from keras.models import Model
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing import image

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from glob import glob

image_files = glob('../datasets/101_ObjectCategories/*/*.jp*g')
image_files += glob('../datasets/256_ObjectCategories/*/*.jp*g')

plt.imshow(image.load_img(np.random.choice(image_files)))
plt.show()

# add preprocessing layer to the front of VGG
resnet = ResNet50(input_shape=(224, 224, 3), weights='imagenet', include_top=True)

resnet.summary()

# make a model to get output before flatten
activation_layer = resnet.get_layer('activation_49')

model = Model(inputs=resnet.inputs, outputs=activation_layer.output)

