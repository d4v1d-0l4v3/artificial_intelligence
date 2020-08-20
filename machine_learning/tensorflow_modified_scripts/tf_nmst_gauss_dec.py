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
from cmath import nan

(train_images, _), (test_images, _) = tf.keras.datasets.mnist.load_data()

def preprocess_images(images):
  images = images.reshape((images.shape[0], 28, 28, 1)) / 255.
  return tf.dtypes.cast(images, tf.float32)
#   return np.where(images > .5, 1.0, 0.0).astype('float32')

train_images = preprocess_images(train_images)
test_images = preprocess_images(test_images)

train_size = 60000
batch_size = 32
test_size = 10000

train_dataset = (tf.data.Dataset.from_tensor_slices(train_images)
                 .shuffle(train_size).batch(batch_size))
test_dataset = (tf.data.Dataset.from_tensor_slices(test_images)
                .shuffle(test_size).batch(batch_size))

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
                filters=2, kernel_size=3, strides=1, padding='same'),
        ]
    )

  @tf.function
  def sample(self, eps=None):
    if eps is None:
      eps = tf.random.normal(shape=(100, self.latent_dim))
    return self.decode(eps, apply_sampling=True)

  def encode(self, x):
    mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
    return mean, logvar

  def reparameterize(self, mean, logvar):
    eps = tf.random.normal(shape=mean.shape)
    return eps * tf.exp(logvar * .5) + mean

  def decode(self, z, apply_sampling=False):
      
    mean, logvar = tf.split(self.decoder(z), num_or_size_splits=2, axis=3)

#     mean = self.decoder(z)
#     logvar = mean 
    
#     logits = self.decoder(z)
    if apply_sampling:
#       sample = self.reparameterize(mean, logvar)
      return mean
  
    return mean, logvar

optimizer = tf.keras.optimizers.Adam(1e-4)


def log_normal_pdf(sample, mean, logvar, raxis=1):
  log2pi = tf.math.log(2. * np.pi)
  
  if raxis == None:
    ret = -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi)
    ret = tf.clip_by_value(ret, clip_value_min=-1e6, clip_value_max=0)
  else:
    ret = tf.reduce_sum(
      -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
      axis=raxis)       
  
  return ret


def compute_loss(model, x):
  mean, logvar = model.encode(x)
  z = model.reparameterize(mean, logvar)
  x_dec_mean, x_dec_logvar = model.decode(z)
#   x_logit = model.decode(z)
#   cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
  logpx_z_tmp = log_normal_pdf(x, x_dec_mean, x_dec_logvar, raxis=None)
  logpx_z = tf.reduce_sum(logpx_z_tmp, axis=[1, 2, 3])
#   tf.print("logpx_z gauss=", logpx_z)
  logpz = log_normal_pdf(z, 0., 0.)
  logqz_x = log_normal_pdf(z, mean, logvar)
#   tf.print ("logpz=", logpz)
  return -tf.reduce_mean(logpx_z + logpz - logqz_x)


@tf.function
def train_step(model, x, optimizer):
  """Executes one training step and returns the loss.

  This function computes the loss and gradients, and uses the latter to
  update the model's parameters.
  """
  with tf.GradientTape() as tape:
    loss = compute_loss(model, x)
    
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

epochs = 20
# set the dimensionality of the latent space to a plane for visualization later
latent_dim = 4
num_examples_to_generate = 16

# keeping the random vector constant for generation (prediction) so
# it will be easier to see the improvement.
random_vector_for_generation = tf.random.normal(
    shape=[num_examples_to_generate, latent_dim])
model = CVAE(latent_dim)

def generate_and_save_images(model, epoch, test_sample):
  mean, logvar = model.encode(test_sample)
  z = model.reparameterize(mean, logvar)
  predictions = model.sample(z)
  fig = plt.figure(figsize=(4, 4))

  for i in range(predictions.shape[0]):
    plt.subplot(4, 4, i + 1)
    plt.imshow(predictions[i, :, :, 0], cmap='gray')
    plt.axis('off')

  # tight_layout minimizes the overlap between 2 sub-plots
  plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
  plt.show()

# Pick a sample of the test set for generating output images

assert batch_size >= num_examples_to_generate
for test_batch in test_dataset.take(1):
  test_sample = test_batch[0:num_examples_to_generate, :, :, :]

generate_and_save_images(model, 0, test_sample)

for epoch in range(1, epochs + 1):
  start_time = time.time()
  for train_x in train_dataset:
    train_step(model, train_x, optimizer)
    
  end_time = time.time()
  loss = tf.keras.metrics.Mean()
  for test_x in test_dataset:
    loss(compute_loss(model, test_x))
    
  elbo = -loss.result()
    
  display.clear_output(wait=False)
  print('****Epoch: {}, Test set ELBO: {}, time elapse for current epoch: {}'
      .format(epoch, elbo, end_time - start_time))
  generate_and_save_images(model, epoch, test_sample)


def display_image(epoch_no):
  return PIL.Image.open('image_at_epoch_{:04d}.png'.format(epoch_no))


plt.imshow(display_image(epoch))
plt.axis('off')  # Display images


anim_file = 'cvae.gif'

with imageio.get_writer(anim_file, mode='I') as writer:
  filenames = glob.glob('image*.png')
  filenames = sorted(filenames)
  last = -1
  for i, filename in enumerate(filenames):
    frame = 2*(i**0.5)
    if round(frame) > round(last):
      last = frame
    else:
      continue
    image = imageio.imread(filename)
    writer.append_data(image)
  image = imageio.imread(filename)
  writer.append_data(image)


import IPython
if IPython.version_info >= (6, 2, 0, ''):
  display.Image(filename=anim_file)


def plot_latent_images(model, n, digit_size=28, latent_dim=2):
  """Plots n x n digit images decoded from the latent space."""

  norm = tfp.distributions.Normal(0, 1)
  grid_x = norm.quantile(np.linspace(0.05, 0.95, n))
  grid_x = tf.reshape(grid_x, [grid_x.shape[0], 1])
  grid_x = tf.broadcast_to(grid_x, [grid_x.shape[0], 
            tf.dtypes.cast(latent_dim, dtype=tf.uint32)])
  grid_y = norm.quantile(np.linspace(0.05, 0.95, n))
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

plot_latent_images(model, 20, 28, latent_dim)
model.save_weights('./vae_weights_latdim8')




