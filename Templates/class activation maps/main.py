from keras.models import Model
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing import image

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from glob import glob

image_files = glob('../datasets/101_ObjectCategories/*/*.jp*g')
image_files += glob('../datasets/256_ObjectCategories/*/*.jp*g')