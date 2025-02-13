# Author: David O
# Date: February 2025
# Retina Fundus Image Pre-training with non-image labels

# imports
import tensorflow as tf
import pandas as pd
from operator import itemgetter
import random
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomZoom
from tensorflow.keras import mixed_precision
from tensorflow.python.client import device_lib
import numpy as np
import gc
import os
# from sklearn.utils.class_weight import compute_class_weight

# Definitions
g_is_kaggle_notebook = True
g_is_google_colab_notebook = False
if g_is_kaggle_notebook:
    retina_base_dir = "/kaggle/input/eyepacs-aptos-messidor-diabetic-retinopathy/augmented_resized_V2"
    imagenet_base_dir = "/kaggle/input/imagenet1k3"
#     in_model_dir = "/kaggle/input/dr_model_cl_5_v1/keras/default/1"
    in_model_dir = '/kaggle/input/best_model_dr_binary_class_2labels/tensorflow2/default/1'
#     in_model_name = "dr_best_class5_model.keras"
    in_model_name = "best_model_dr_binary_class_2labels.h5"
    in_model_file_path = in_model_dir + "/" + in_model_name
#     in_weights_file_path = in_model_dir + "/" + in_model_name
    # Input data files are available in the read-only "../input/" directory
    # For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
    output_model_file_path = 'retina_pretrain_model_v1.keras'
    output_weights_file_path = 'retina_pretrain_model_v1.weights.h5'
    # You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
    # You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
elif g_is_google_colab_notebook:
    usr_drive_dir = "/content/drive/MyDrive/ai_datasets/diabetic-retinopathy"
    # !unzip $usr_drive_dir"/augmented_resized_V2.zip" -d /content/sample_data/
    retina_base_dir = "/content/sample_data"
    output_model_file_path = usr_drive_dir + '/retina_pretrain_model_v1.h5'
else:
    print ("Error, notebook not supported")
    quit()

cache_dir = retina_base_dir + '/cache'
retina_train_dir = retina_base_dir + "/train"
imagenet_train_dir = imagenet_base_dir 
test_dir = retina_base_dir + "/test"
val_dir = retina_base_dir + "/val"
g_img_height_pixels = 600
g_img_width_pixels = 600

g_input_channels = 3  # RGB pixels
g_num_classes = 1
g_training_epochs = 24
g_load_model = False
g_fine_tuning_trainable_layers = 14
latent_dim = 40
AUTOTUNE = tf.data.AUTOTUNE
avail_gpus = len(tf.config.experimental.list_physical_devices('GPU'))
avail_tpus = len(tf.config.experimental.list_physical_devices('TPU'))
parallel_proc_units = 1
g_base_model_name = "resnet152"


# Prioritize GPUs
if avail_gpus > 0:
    strategy = tf.distribute.get_strategy()
    parallel_proc_units = 1
    # parallel_proc_units = avail_gpus
    # Manage distributed processing strategy
    # strategy = tf.distribute.MirroredStrategy()
elif avail_tpus > 0:
    parallel_proc_units = avail_tpus
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver()  # Automatically detects TPU on Kaggle
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.TPUStrategy(resolver)
else:
    strategy = tf.distribute.get_strategy()

# strategy = tf.distribute.get_strategy()

g_batch_size = int(32 * parallel_proc_units)


print(device_lib.list_local_devices())
print("Num GPUs Available: ", avail_gpus)
print("Num TPUs Available: ", avail_tpus)
print("Tensorflow version=", tf.__version__)

# Enable mixed precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# Get list of all image file paths recursively
retina_files = tf.data.Dataset.list_files(os.path.join(retina_train_dir, "*/*"), shuffle=False)  # Subdirectories assumed
imagenet_files = tf.data.Dataset.list_files(os.path.join(imagenet_train_dir, "*/*"), shuffle=False)

# Combine datasets
train_ds = retina_files.concatenate(imagenet_files)

# Shuffle the dataset. Size of buffer is set to all elements since initially the dataset is a filename dataset.
# Currently, do not reshuffle on each iteration because the dataset will contain images
train_ds = train_ds.shuffle(buffer_size=train_ds.cardinality(), reshuffle_each_iteration=False)
# Assumes folder names of retina classes are ["0", "1", "2", "3", "4"] and other non-retina
# classes are named "500" to "999"
max_retinal_label_num=4

# Function to load and preprocess image
@tf.function
def load_image(filepath):
    image = tf.io.read_file(filepath)
    image = tf.image.decode_jpeg(image, channels=3)  # Adjust format (JPEG/PNG) as needed
    image = tf.image.resize(image, [g_img_height_pixels, g_img_width_pixels])  # Resize to model input size
    # image = image / 255.0  # Normalize
    label_str = tf.strings.split(filepath, os.sep)[-2]  # Get subdirectory name as label
    # For binary cross entropy cast label to float type
    label_num = tf.strings.to_number(label_str, out_type=tf.float32)
    if label_num > max_retinal_label_num:
        label_num = 0
    else:
        label_num = 1
        
    return image, label_num

# Apply mapping
train_ds = train_ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)

# Batch & prefetch for performance
train_ds = train_ds.batch(g_batch_size).prefetch(tf.data.AUTOTUNE)


# Preprocessed training images and combined with preprocessed labels
# retina_train_ds = tf.keras.utils.image_dataset_from_directory(
#     train_dir,
#     labels="inferred",
#     label_mode="int",
#     color_mode="rgb",
#     # class_names=["0", "1", "2", "3", "4"],
#     subset=None,
#     seed=random.randint(0, 512000),
#     image_size=(g_img_height_pixels, g_img_width_pixels),
#     batch_size=g_batch_size,
#     shuffle=True,
#     validation_split=None,
#     crop_to_aspect_ratio=False
#     # Keep as much info without distortion with the padding caveat
#     # pad_to_aspect_ratio=True
# )
#
# # Set all retina files to one class for pretrained network to detect retina images
# retina_train_ds.map(lambda image, label: image, label = 0)
#
# classes_array = np.arange(g_num_classes)
#
# # Preprocessed training images and combined with preprocessed labels
# imagenet_train_ds = tf.keras.utils.image_dataset_from_directory(
#     imagenet_train_dir,
#     labels="inferred",
#     label_mode="int",
#     color_mode="rgb",
#     # class_names=["0", "1", "2", "3", "4"],
#     subset=None,
#     seed=random.randint(0, 512000),
#     image_size=(g_img_height_pixels, g_img_width_pixels),
#     batch_size=g_batch_size,
#     shuffle=True,
#     validation_split=None,
#     crop_to_aspect_ratio=False
#     # Attempt to match specified image size even if it aspect ratio needs to be modified
#     pad_to_aspect_ratio=False
# )
#
# # Set all non-retina files to one class (with different label to retina images class label) for pre-trained 
# # network to detect retina images.
# imagenet_train_ds.map(lambda image, label: image, label = 1)


# Test delete
# train_ds = train_ds.filter(filter_by_label).map(map_to_consolidate_desease_level).prefetch(buffer_size=AUTOTUNE)
# print("elements in dataset=", train_ds.cardinality(), "unknown=", tf.data.UNKNOWN_CARDINALITY)
#
# test_ds = test_ds.batch(1).prefetch(buffer_size=AUTOTUNE)
# # train_imgs = []
# # train_labels = []
# no_dr_ctr = 0
# dr1_ctr = 0
# dr2_ctr = 0
# dr3_ctr = 0
# dr4_ctr = 0
# other_dr_ctr = 0
# for element in test_ds:
#     # print("Processing batch", element)
#     image, label = element
#     # print("label numpy=", label.numpy())
#     if label.numpy() == 0:
#         no_dr_ctr = no_dr_ctr + 1
#     elif label.numpy() == 1:
#         dr1_ctr = dr1_ctr + 1
#     elif label.numpy() == 2:
#         dr2_ctr = dr2_ctr + 1
#     elif label.numpy() == 3:
#         dr3_ctr = dr3_ctr + 1
#     elif label.numpy() == 4:
#         dr4_ctr = dr4_ctr + 1
#     else:
#         other_dr_ctr = other_dr_ctr + 1
#
# # Delete the dataset and collect garbage
# del test_ds
# gc.collect()
#
# print("*** End of Filtering images no dr ctr=", no_dr_ctr, " dr1 ctr=", dr1_ctr,
#       " dr2 ctr=", dr2_ctr, " dr3 ctr=", dr3_ctr, " dr4 ctr=", dr4_ctr,
#       "other_dr_ctr=", other_dr_ctr)
#
#
# # Test delete!!!
# quit()

# Data augmentation
data_augmentation = tf.keras.Sequential([
    RandomFlip("horizontal")
])

# retina_train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)

# val_ds = val_ds.take(1).filter(filter_by_label).map(map_to_consolidate_desease_level).batch(1).prefetch(buffer_size=AUTOTUNE)
# val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

with strategy.scope():
    if g_load_model:
        base_weights = None
    else:
        base_weights = 'imagenet'

    base_model = tf.keras.applications.ResNet152(
          input_shape=(g_img_height_pixels, g_img_width_pixels, g_input_channels),
          include_top=False
          ,
          weights=base_weights)        

    inputs = tf.keras.Input(shape=(g_img_height_pixels, g_img_width_pixels, g_input_channels), dtype="uint8")
    if g_is_kaggle_notebook:
        x = tf.keras.ops.cast(inputs, "float32")
    elif g_is_google_colab_notebook:
        x = tf.cast(inputs, tf.float32)
    else:
        print ("Unsupported notebook")
        quit()

    x = tf.keras.applications.resnet.preprocess_input(x)
    # x = tf.keras.layers.Rescaling(1. / 255,
    #                               input_shape=(g_img_height_pixels, g_img_width_pixels, g_input_channels))(inputs)
    # x = data_augmentation(x)
    x = base_model(x, training=False)
    # x = tf.keras.layers.Conv2D(
    #         filters=64, kernel_size=6, strides=(2, 2), activation='relu', kernel_regularizer=l2(0.0005))(x)
    # x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=l2(0.0005))(x)
    # x = tf.keras.layers.BatchNormalization()(x, training=False)
    # x = tf.keras.layers.Dropout(0.5)(x)
    # outputs = tf.keras.layers.Dense(g_num_classes, activation='softmax')(x)
    outputs = tf.keras.layers.Dense(g_num_classes, dtype='float32', activation='sigmoid')(x)
    model = tf.keras.Model(inputs, outputs)
    
    # Model with pre-trained base
    if g_load_model:
        # print ("User loading model from file:", in_model_file_path)
        # model = tf.keras.models.load_model (in_model_file_path)
        print ("User loading model weights from file:", in_model_file_path)
        model.load_weights (in_model_file_path)
        

# @tf.function
# def reset_allowed_no_dr_samples_counter(img, lbl):
#     global allowed_no_dr_samples_counter
#     allowed_no_dr_samples_counter = allowed_no_dr_samples_counter.assign(0)
#     with tf.control_dependencies([allowed_no_dr_samples_counter]):
#         return img, lbl

# Rest fitter for 'no DR' labeled element
class MyCallback(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs=None):
        global allowed_no_dr_samples_counter
        # allowed_no_dr_samples_counter = 0
        keys = list(logs.keys())
        tf.print("******End epoch {} of training; got log keys: {} *************** ".format(epoch, keys))
        tf.print ("allowed_no_dr_samples_counter=", allowed_no_dr_samples_counter)


class NaNChecker(tf.keras.callbacks.Callback):
    def on_batch_end(self, batch, logs=None):
        if np.isnan(logs.get('loss')):
            print(f'NaN detected at batch {batch}')
            self.model.stop_training = True


# Callbacks
callbacks = [
    NaNChecker(),
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6),
    tf.keras.callbacks.ModelCheckpoint(output_weights_file_path, monitor='val_loss', 
                                       save_best_only=True, save_weights_only=True)
]

model.trainable = True
# base_model.summary(show_trainable=True)
if not g_load_model:
    print("**************Training with frozen base model**************")
    base_model.trainable = False  # Freeze the base model
    tuned_learning_rate = 0.001 / parallel_proc_units
    # Test delete
    base_model.summary(show_trainable=True)

    with strategy.scope():
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=tuned_learning_rate),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.5, name='accuracy')])


        history = model.fit(
            train_ds,
            # validation_data=val_ds,
            validation_data=train_ds, # Modify!!!
#           epochs=g_training_epochs,
            epochs=6,
            callbacks=callbacks
        )

        
model.trainable = True
# Enable training on all layers except in base pretrained model (trainable
# setting already handle above)
base_model.trainable = True
for layer in base_model.layers[0:-g_fine_tuning_trainable_layers]:
     layer.trainable = False
        
print ("*******Fine tuning******")
base_model.summary(show_trainable=True)

tuned_learning_rate = 1e-5 / parallel_proc_units # Low learning rate
# tuned_learning_rate = 0.001 / parallel_proc_units # Low learning rate
# after_base_model_batch_reg (after_base_model_batch_o, training=False)

with strategy.scope():
    # tuned_learning_rate = 0.001
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(tuned_learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.5, name='accuracy')]
    )

    history = model.fit(
        train_ds,
        # validation_data=val_ds,
        validation_data=train_ds, # Modify!!!
        epochs=g_training_epochs,
        callbacks=callbacks
    )

del train_ds
del val_ds
gc.collect()

# Evaluate model

# Preprocessed training images and combined with preprocessed labels
# test_ds = tf.keras.utils.image_dataset_from_directory(
#     test_dir,
#     labels="inferred",
#     label_mode="int",
#     color_mode="rgb",
#     class_names=["0", "1", "2", "3", "4"],
#     subset=None,
#     seed=random.randint(0, 512000),
#     image_size=(g_img_height_pixels, g_img_width_pixels),
#     batch_size=g_batch_size,
#     shuffle=False,
#     validation_split=None,
#     crop_to_aspect_ratio=False
    # Keep as much info without distortion with the padding caveat
    # pad_to_aspect_ratio=True
#)

# model.trainable = False  # Freeze the base model
# # model.summary(show_trainable=True)
# print("**************************************")
# print("********** Testing Model *************")
# print ("*************************************")
#
# with strategy.scope():
#     model.compile(
#         optimizer=tf.keras.optimizers.Adam(tuned_learning_rate),
#         loss=tf.keras.losses.SparseCategoricalCrossentropy(),
#         metrics=['accuracy'],
#     )
#
#     model.evaluate(
#     x=test_ds,
#     verbose='auto',
#     sample_weight=None,
#     steps=None,
#     callbacks=None,
#     return_dict=False,
#     )
