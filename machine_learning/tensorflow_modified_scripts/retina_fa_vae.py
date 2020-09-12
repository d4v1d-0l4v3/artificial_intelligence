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
#from tensorflow_large_model_support import LMS
import tensorflow_probability as tfp

#tf.config.experimental.set_lms_enabled(True)
# lms_obj = LMS()
# lms_obj.run(tf.compat.v1.get_default_graph())

# Constants
images_home_dir = '/mnt/media/davidolave/My Passport/datasets/retina/diabetic-retinopathy-detection/test_main'
img_height_pixels = 128 # 3168
img_width_pixels = 128 # 4752
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
    
  
assert g_batches > MIN_BATCHES
assert g_train_batches > MIN_BATCHES
assert g_test_batches > MIN_BATCHES

img_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

# Creates image iterator
image_itr = img_gen.flow_from_directory(directory=images_home_dir, 
        target_size=(img_height_pixels, img_width_pixels), 
        batch_size=g_batch_size, shuffle=True,
        class_mode=None)
        
# list_ds = tf.data.Dataset.from_generator(
#     img_gen.flow_from_directory, args=(images_home_dir, 
#     (img_height_pixels, img_width_pixels)), 
#     output_types=(tf.float32), 
#     output_shapes=([imgs_to_load, img_height_pixels, img_width_pixels, g_input_channels])
# )

image_count = imgs_to_load

'''
data_dir = pathlib.Path(images_home_dir)
image_count = len(list(data_dir.glob('*.jpeg')))

# Create datasets
list_ds = tf.data.Dataset.list_files(str(data_dir/'*.jpeg'), shuffle=True)
list_ds = list_ds.shuffle(image_count, reshuffle_each_iteration=True)
'''

# Warning: Always shuffle for improved generation quality
#list_ds.shuffle(image_count, reshuffle_each_iteration=True)
'''
val_size = int(image_count * 0.2)
train_ds = list_ds.skip(val_size)
val_ds = list_ds.take(val_size)
'''

def decode_img(img, input_channels = 3):
  # convert the compressed string to a 3D uint8 tensor
  img = tf.image.decode_jpeg(img, channels=input_channels)
  # resize the image to the desired size
  return tf.image.resize(img, [img_height_pixels, img_width_pixels])
  # return img

def preprocess_image(image):
  image = image / 255.
  return tf.dtypes.cast(image, tf.float32)

def process_path(file_path):
  # load the raw data from the file as a string
  img = tf.io.read_file(file_path)
  img = decode_img(img)
  img = preprocess_image (img)
  
  return img

# Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
'''
train_dataset = train_ds.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
test_dataset = val_ds.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
'''

# for image, label in train_ds.take(-1):
#   print("Image shape: ", image.numpy().shape)
#   print("Label: ", label.numpy())
#   print ("pixel sample", image [1000, 1000, :])
'''
train_dataset = train_dataset.batch(g_batch_size)
test_dataset = test_dataset.batch(g_batch_size)
'''

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
            tf.keras.layers.Conv2D(
                filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
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
            tf.keras.layers.Dense(units= (img_height_pixels >> 2) * (img_width_pixels >> 2) * 16,
            #tf.keras.layers.Dense(units= (img_height_pixels >> 4) * (img_width_pixels >> 4) * 4,
               activation=tf.nn.relu),
            tf.keras.layers.Reshape(target_shape=((img_height_pixels >> 2),
                (img_width_pixels >> 2), 16)),
                # (img_width_pixels >> 4), 4)),
            tf.keras.layers.Conv2DTranspose(
                # filters=64, kernel_size=3, strides=2, padding='same',
                filters=32, kernel_size=8, strides=2, padding='same',
                activation='relu'),
            tf.keras.layers.Conv2DTranspose(
                # filters=32, kernel_size=3, strides=2, padding='same',
                filters=16, kernel_size=8, strides=2, padding='same',
                activation='relu'),
            tf.keras.layers.Conv2DTranspose(
                filters=4, kernel_size=3, strides=1, padding='same',
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

  @tf.function
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
  mean = tf.dtypes.cast(mean, dtype=tf.float32)
  logvar = tf.dtypes.cast(logvar, dtype=tf.float32)
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

epochs = 120
# set the dimensionality of the latent space to a plane for visualization later
latent_dim = 10
num_examples_to_generate = 2

# keeping the random vector constant for generation (prediction) so
# it will be easier to see the improvement.
random_vector_for_generation = tf.random.normal(
    shape=[num_examples_to_generate, latent_dim])
model = CVAE(latent_dim, g_input_channels)

# sess = tf.keras.backend.get_session()
# init = tf.compat.v1.global_variables_initializer()
# sess.run(init)
# from tensorflow.python.keras.backend import set_session
# set_session(sess)


def generate_and_save_images(model, epoch, test_sample):
  with options({'memory': True}):  
    mean, logvar = model.encode(test_sample)
    z = model.reparameterize(mean, logvar)
    predictions = model.sample(z)
  
  display_images = test_sample.shape[0]
  if display_images <= 0:
    display_images = 1
  
  fig = plt.figure(figsize=(display_images, display_images))

  for i in range(predictions.shape[0]):    
    plt.subplot(display_images, display_images, i + 1)
    #with tf.Session() as sess_tmp:
    #pred_array = predictions[i, :, :, :]
    #pred_array = pred_array.eval()
    #plt.imshow(pred_array)
    plt.imshow(predictions[i, :, :, :])
    plt.axis('off')

  # tight_layout minimizes the overlap between 2 sub-plots
  plt.savefig('image_at_epoch_{:04d}.jpg'.format(epoch))
  plt.show()
  plt.close(fig)
  
  fig = plt.figure(figsize=(display_images, display_images))

  for i in range(test_sample.shape[0]):    
    plt.subplot(display_images, display_images, i + 1)
    plt.imshow(test_sample[i, :, :, :])
    plt.axis('off')

  # tight_layout minimizes the overlap between 2 sub-plots
  plt.savefig('ref_image_at_epoch_{:04d}.jpg'.format(epoch))
  plt.show()
  plt.close(fig)

# Pick a sample of the test set for generating output images
assert g_batch_size >= num_examples_to_generate

'''
for test_batch in test_dataset.take(1):  
  test_sample = test_batch[0:num_examples_to_generate, :, :, :]
'''

def display_image(epoch_no):
  return PIL.Image.open('image_at_epoch_{:04d}.jpg'.format(epoch_no))

#@tf.function
def train_op ():
    test_sample = next(image_itr)
    generate_and_save_images(model, 0, test_sample)

    for epoch in range(1, epochs + 1):
      start_time = time.time()
      '''
      for train_x in train_dataset:
        train_step(model, train_x, optimizer)
      '''
      image_itr.reset()  
        
      for step in range(g_train_batches):
        # with tf.profiler.experimental.Trace('train', step_num=step, _r=1):
        batch = next(image_itr)
        train_step(model, batch, optimizer)
        
      end_time = time.time()
      loss = tf.keras.metrics.Mean()
      '''
      for test_x in test_dataset:
        loss(compute_loss(model, test_x))
      '''
      for i in range(g_test_batches):
        batch = next(image_itr)  
        loss(compute_loss(model, batch))
          
      elbo = -loss.result()
        
      display.clear_output(wait=False)
      tf.print('****Epoch: {}, Test set ELBO: {}, time elapse for current epoch: {}'
          .format(epoch, elbo, end_time - start_time))
      generate_and_save_images(model, epoch, test_sample)
      
    plt.imshow(display_image(epoch))
    plt.axis('off')  # Display images  

# tf.profiler.experimental.start('/mnt/home/davidolave/git/artificial_intelligence/machine_learning/',
#     options = tf.profiler.experimental.ProfilerOptions(
#               host_tracer_level=3, python_tracer_level=1, device_tracer_level=1))
#with tf.Session() as sess:
train_op()
# tf.profiler.experimental.stop()


# with sv.managed_session() as sess:
  # Train normally
#   sess.run(train_op)

anim_file = 'cvae.gif'

with imageio.get_writer(anim_file, mode='I') as writer:
  filenames = glob.glob('image*.jpg')
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
  
def plot_latent_images(model, n, img_height = img_height_pixels,
                        img_width = img_width_pixels, latent_dim=2,
                        channels = 3):
  """Plots n x n digit images decoded from the latent space."""

  norm = tfp.distributions.Normal(0, 1)
  grid_x = norm.quantile(np.linspace(0.05, 0.95, n))
  # grid_x = numpy.random.normal(0, 1, n)
  grid_x = tf.reshape(grid_x, [grid_x.shape[0], 1])
  grid_x = tf.broadcast_to(grid_x, [grid_x.shape[0], 
            tf.dtypes.cast(latent_dim, dtype=tf.uint32)])
  grid_y = norm.quantile(np.linspace(0.05, 0.95, n))
  # grid_y = numpy.random.normal(0, 1, n)
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

  plt.figure(figsize=(10, 10))
  plt.imshow(image_set)
  plt.axis('Off')
  plt.show()
  plt.savefig('image_manifold.jpg')

print ("Update****************")
plot_latent_images(model, 2, img_height_pixels, 
                   img_width_pixels, latent_dim)
model.save_weights('./retina_vae_weights')
