from IPython import display

import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
import os
import time

class CVAE(tf.keras.Model):
  """Convolutional variational autoencoder."""

  def __init__(self, latent_dim):
    super(CVAE, self).__init__()
    self.latent_dim = latent_dim
    self.encoder = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(
                filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
            tf.keras.layers.Conv2D(
                filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
            tf.keras.layers.Flatten(),
            # No activation
            tf.keras.layers.Dense(latent_dim + latent_dim),
        ]
    )

    self.decoder = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
            tf.keras.layers.Dense(units=7*7*32, activation=tf.nn.relu),
            tf.keras.layers.Reshape(target_shape=(7, 7, 32)),
            tf.keras.layers.Conv2DTranspose(
                filters=64, kernel_size=3, strides=2, padding='same',
                activation='relu'),
            tf.keras.layers.Conv2DTranspose(
                filters=32, kernel_size=3, strides=2, padding='same',
                activation='relu'),
            # No activation
            tf.keras.layers.Conv2DTranspose(
                filters=1, kernel_size=3, strides=1, padding='same'),
        ]
    )

  @tf.function
  def sample(self, eps=None):
    if eps is None:
      eps = tf.random.normal(shape=(100, self.latent_dim))
    return self.decode(eps, apply_sigmoid=True)

  def encode(self, x):
    mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
    return mean, logvar

  def reparameterize(self, mean, logvar):
    eps = tf.random.normal(shape=mean.shape)
    return eps * tf.exp(logvar * .5) + mean

  def decode(self, z, apply_sigmoid=False):
    logits = self.decoder(z)
    if apply_sigmoid:
      probs = tf.sigmoid(logits)
      return probs
    return logits

def plot_latent_images(model, n, digit_size=28, latent_dim=2):
  """Plots n x n digit images decoded from the latent space."""

  norm = tfp.distributions.Normal(0, 1)
#   grid_x = norm.quantile(np.linspace(0.05, 0.95, n))
  grid_x = norm.sample(n)
  grid_x = tf.reshape(grid_x, [grid_x.shape[0], 1])
  grid_x = tf.broadcast_to(grid_x, [grid_x.shape[0], 
            tf.dtypes.cast(latent_dim, dtype=tf.uint32)])
#   grid_y = norm.quantile(np.linspace(0.05, 0.95, n))
  grid_y = norm.sample(n)
  grid_y = tf.reshape(grid_y, [grid_y.shape[0], 1])
  grid_y = tf.broadcast_to(grid_y, [grid_y.shape[0],
            tf.dtypes.cast(latent_dim, dtype=tf.uint32)])
  image_width = digit_size*n
  image_height = image_width
  image = np.zeros((image_height, image_width))

  for i in range(len(grid_x)):
    for j in range(len(grid_y)):
      z = np.array([grid_x[i, : ], grid_y[j, : ]])
#       z = tf.reshape(z, [-1])
      x_decoded = model.sample(z)
#       print ("shape:", x_decoded.shape)
      digit = tf.reshape(x_decoded[0], (digit_size, digit_size))
      image[i * digit_size: (i + 1) * digit_size,
        j * digit_size: (j + 1) * digit_size] = digit.numpy()
        
  plt.figure(figsize=(10, 10))
  plt.imshow(image, cmap='Greys_r')
  plt.axis('Off')
  plt.show()
  plt.savefig('image_manifold.png')

# set the dimensionality of the latent space to a plane for visualization later
latent_dim = 2

model = CVAE(latent_dim)
model.load_weights('./vae_weights_latdim8')

plot_latent_images(model, 20, 28, model.latent_dim) 

