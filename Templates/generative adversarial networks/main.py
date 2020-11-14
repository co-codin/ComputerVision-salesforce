from tensorflow.keras.layers import Input, Dense, LeakyReLU, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys, os

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# map inputs to (-1, +1) for better training
x_train, x_test = x_train / 255.0 * 2 - 1, x_test / 255.0 * 2 - 1
print("x_train.shape: ", x_train.shape)

N, H, W = x_train.shape
D = H * W

x_train, x_test = x_train.reshape(-1, D), x_test.reshape(-1, D)

print("x_train.shape: ", x_train.shape)

latent_dim = 100

def build_generator(latent_dim):
    i = Input(shape=(latent_dim,))
    x = Dense(256, activation=LeakyReLU(alpha=0.2))(i)
    x = BatchNormalization(momentum=0.8)(x)
    x = Dense(512, activation=LeakyReLU(alpha=0.2))(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Dense(1024, activation=LeakyReLU(alpha=0.2))(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Dense(D, activation='tanh')(x)

    model = Model(i, x)
    return model

def build_discriminator(img_size):
    i = Input(shape=(img_size,))
    x = Dense(512, activation=LeakyReLU(alpha=0.2))(i)
    x = Dense(256, activation=LeakyReLU(alpha=0.2))(x)
    x = Dense(1, activation='sigmoid')(x)
    model = Model(i, x)
    return model


# Compile both models in preparation for training
discriminator = build_discriminator(D)
discriminator.compile(
    loss='binary_crossentropy',
    optimizer=Adam(0.0002, 0.5),
    metrics=['accuracy']
)

# create an input to represent noise sample from latent space
generator = build_generator(latent_dim)

# create an input to represent noise sample from latent space
z = Input(shape=(latent_dim,))

# pass  noise through generator to get an image
img = generator(z)

# make sure only the generator is trained
discriminator.trainable = False

# the true output is fake, but we label them real
fake_pred = discriminator(img)

# create the combined model object
combined_model = Model(z, fake_pred)

combined_model.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))

batch_size = 32
epochs = 30000
sample_period = 200 # every `sample_period` steps generate and save some data

# create batch labels to use when calling train_on_batch
ones = np.ones(batch_size)
zeros = np.zeros(batch_size)

# store the losses
d_losses = []
g_losses = []

# create a folder to store generated images
if not os.path.exists('gan_images'):
    os.makedirs('gan_images')

# A function to generate a grid of random samples from generator and save them to file
def sample_images(epoch):
    rows, cols = 5, 5
    noise = np.random.randn(rows * cols, latent_dim)
    imgs = generator.predict(noise)

    # Rescale images 0 - 1
    imgs = 0.5 * imgs + 0.5

    fig, axs = plt.subplots(rows, cols)
    idx = 0
    for i in range(rows):
        for j in range(cols):
            axs[i, j].imshow(imgs[idx].reshape(H, W), cmap='gray')
            axs[i, j].axis('off')
            idx += 1
        fig.savefig("gan_images/%d.png" % epoch)
        plt.close()