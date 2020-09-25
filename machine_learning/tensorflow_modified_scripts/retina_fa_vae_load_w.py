'''
This file implements fa retina generation by VAE design
'''
from IPython import display
import pathlib
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import os
import PIL
import PIL.Image
import matplotlib.pyplot as plt
import time
import imageio
import glob
import contextlib

#import tensorflow_probability as tfp

# Constants
os.environ['TF_GPU_HOST_MEM_LIMIT_IN_MB'] = '12000'
tf.config.experimental.set_lms_enabled(True)    
tf.config.experimental.set_lms_defrag_enabled(True) 

# Constants
images_home_dir = '/media/davidolave/My Passport/datasets/retina/diabetic-retinopathy-detection/test_main'
img_height_pixels = 512 # 3168
img_width_pixels = 512 # 4752
g_input_channels = 3  # RGB pixels
g_batch_size = 2
imgs_to_load = 160.
g_batches_tf = imgs_to_load / tf.dtypes.cast (g_batch_size, tf.float32)
g_train_batches_tf = g_batches_tf * 0.8
g_test_batches_tf = g_batches_tf * 0.2
g_train_batches_tf = tf.dtypes.cast(g_train_batches_tf, tf.uint32)
g_test_batches_tf = tf.dtypes.cast(g_test_batches_tf, tf.uint32)
g_batches_tf = tf.dtypes.cast(g_batches_tf, tf.uint32)
g_batches = 0
g_test_batches = 0
g_train_batches = 0

MIN_BATCHES = 1 # Min allowed batches number

# Value sanity checks
#with tf.Session() as sess_check:
g_batches = g_batches_tf
g_test_batches = g_test_batches_tf
g_train_batches = g_train_batches_tf
    
  
# assert g_batches > MIN_BATCHES
# assert g_train_batches > MIN_BATCHES
# assert g_test_batches > MIN_BATCHES

img_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

# Creates image iterator
image_itr = img_gen.flow_from_directory(directory=images_home_dir, 
        target_size=(img_height_pixels, img_width_pixels), 
        batch_size=g_batch_size, shuffle=True,
        class_mode=None)

class CVAE(tf.keras.Model):
  """Convolutional variational autoencoder."""

  def __init__(self, latent_dim, input_channels = 3):
    super(CVAE, self).__init__()
    self.latent_dim = latent_dim
    self.encoder = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(img_height_pixels, img_width_pixels, 
                input_channels)),
            tf.keras.layers.Conv2D(
                filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
                # filters=8, kernel_size=8, strides=(4, 4), activation='relu'),
            tf.keras.layers.Conv2D(
                filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
                # filters=8, kernel_size=8, strides=(4, 4), activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), 
                padding='same'),
            
            tf.keras.layers.Conv2D(
                filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
            
            tf.keras.layers.MaxPooling2D(pool_size=(5, 5), strides=(1, 1), 
                padding='same'),
            
            tf.keras.layers.Flatten(),            
            # No activation
            tf.keras.layers.Dense(latent_dim + latent_dim),
            tf.keras.layers.Dense(latent_dim + latent_dim)
        ]
    )

    self.decoder = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
            tf.keras.layers.Dense(units= latent_dim, activation=tf.nn.relu),
            tf.keras.layers.Dense(units= (img_height_pixels >> 2) * 
                        (img_width_pixels >> 2) * 64,
            #tf.keras.layers.Dense(units= (img_height_pixels >> 4) * (img_width_pixels >> 4) * 4,
               activation=tf.nn.relu),
            tf.keras.layers.Reshape(target_shape=((img_height_pixels >> 2),
                (img_width_pixels >> 2), 64)),
                # (img_width_pixels >> 4), 4)),
            tf.keras.layers.Conv2DTranspose(
                # filters=64, kernel_size=3, strides=2, padding='same',
                filters=64, kernel_size=8, strides=2, padding='same',
                activation='relu'),
            tf.keras.layers.Conv2DTranspose(
                # filters=32, kernel_size=3, strides=2, padding='same',
                filters=64, kernel_size=8, strides=2, padding='same',
                activation='relu'),
            tf.keras.layers.Conv2DTranspose(
                filters=32, kernel_size=3, strides=1, padding='same',
                activation='relu'),
            # No activation
            tf.keras.layers.Conv2DTranspose(
                filters=2*input_channels, kernel_size=3, strides=1, padding='same'),
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

# set the dimensionality of the latent space to a plane for visualization later
latent_dim = 40

@contextlib.contextmanager
def options(options):
  old_opts = tf.config.optimizer.get_experimental_options()
  tf.config.optimizer.set_experimental_options(options)
  try:
    yield
  finally:
    tf.config.optimizer.set_experimental_options(old_opts)

old_opts = tf.config.optimizer.get_experimental_options()
print ("********Old Options**********", old_opts)

model = CVAE(latent_dim, g_input_channels)

def generate_and_save_images(model, epoch, test_sample):
  with options({'memory': True}):  
    mean, logvar = model.encode(test_sample)
    z = model.reparameterize(mean, logvar)
    predictions = model.sample(z)
  
  display_images = test_sample.shape[0]
  if display_images <= 0:
    display_images = 1
  
  h_mgs_no = display_images  # Hieght in images
  w_mgs_no = display_images  # Width in images
  fig = plt.figure(figsize=(h_mgs_no, w_mgs_no))

  for i in range(predictions.shape[0]):    
    plt.subplot(h_mgs_no, w_mgs_no, i + 1)
    #with tf.Session() as sess_tmp:
    #pred_array = predictions[i, :, :, :]
    #pred_array = pred_array.eval()
    #plt.imshow(pred_array)
    plt.imshow(predictions[i, :, :, :])
    plt.axis('off')

  # tight_layout minimizes the overlap between 2 sub-plots
  plt.savefig('image_at_epoch_{:04d}.jpg'.format(epoch))
  #plt.show()
  plt.close(fig)
  
  fig = plt.figure(figsize=(h_mgs_no, w_mgs_no))

  for i in range(test_sample.shape[0]):    
    plt.subplot(h_mgs_no, w_mgs_no, i + 1)
    plt.imshow(test_sample[i, :, :, :])
    plt.axis('off')

  # tight_layout minimizes the overlap between 2 sub-plots
  plt.savefig('ref_image_at_epoch_{:04d}.jpg'.format(epoch))
  #    plt.show()
  plt.close(fig)
  
def plot_latent_images(model, n, img_height = img_height_pixels,
                        img_width = img_width_pixels, latent_dim=2,
                        channels = 3):
  """Plots n x n digit images decoded from the latent space."""

  #norm = tfp.distributions.Normal(0, 1)
  # grid_x = norm.quantile(np.linspace(0.05, 0.95, n))
  # grid_x = norm.sample(n)
  grid_x = np.random.normal(0, 1, n)
  grid_x = tf.reshape(grid_x, [grid_x.shape[0], 1])
  grid_x = tf.broadcast_to(grid_x, [grid_x.shape[0], 
            tf.dtypes.cast(latent_dim, dtype=tf.uint32)])
  # grid_y = norm.quantile(np.linspace(0.05, 0.95, n))
  grid_y = np.random.normal(0, 1, n)
  #grid_y = norm.sample(n)
  grid_y = tf.reshape(grid_y, [grid_y.shape[0], 1])
  grid_y = tf.broadcast_to(grid_y, [grid_y.shape[0],
            tf.dtypes.cast(latent_dim, dtype=tf.uint32)])
  images_width = img_width*n
  images_height = img_height*n
  image_set = np.zeros((images_height, images_width, channels))

  for i in range(len(grid_x)):
    for j in range(len(grid_y)):
      z = np.array([grid_x[i, : ], grid_y[j, : ]])
#       z = tf.reshape(z, [-1])
      x_decoded = model.sample(z)
#       print ("shape:", x_decoded.shape)
      digit = tf.reshape(x_decoded[0], (img_height, img_width, channels))
      image_set[i * img_height: (i + 1) * img_height,
        j * img_width: (j + 1) * img_width, :] = digit.numpy()

  plt.figure(figsize=(n, n))
  plt.imshow(image_set)
  plt.axis('Off')
  plt.show()
  plt.savefig('image_manifold.jpg')

model.load_weights('./retin_vae_w_lat_dim40_epochs400')
for i in range(g_train_batches):
    plot_latent_images(model, 2, img_height_pixels, 
                   img_width_pixels, latent_dim)

# for i in range(g_train_batches):
#     test_sample = next(image_itr)
#     generate_and_save_images(model, i, test_sample)
    
    
