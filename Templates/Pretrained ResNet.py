from __future__ import print_function, division
from builtins import range, input
# Note: you may need to update your version of future
# sudo pip install -U future

from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.resnet50 import ResNet50, preprocess_input
# from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator

from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

from glob import glob

# resize all the images to this
IMAGE_SIZE = [224, 224]

epochs = 16
batch_size = 32

# https://www.kaggle.com/paultimothymooney/blood-cells
train_path = './datasets/blood_cell/images/TRAIN'
valid_path = './datasets/blood_cell/images/TEST'

# get number of files
image_files = glob(train_path + '/*/*.jp*g')
valid_image_files = glob(valid_path + '/*/*.jp*g')

# get number of classes
folders = glob(train_path + '/*')

# plt.imshow(image.load_img(np.random.choice(image_files)))
# plt.show()