# Author: David O
# Date: July 2024
# Classification of diabetic-retinopathy presence levels

# imports
import tensorflow as tf
import pandas as pd
from operator import itemgetter
import random
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import RandomFlip, RandomRotation, RandomZoom
from tensorflow.keras import mixed_precision
import numpy as np
import gc
from sklearn.utils.class_weight import compute_class_weight

# Definitions
base_dir = "/media/davidolave/OS/Downloads/ai_datasets/diabetic-retinopathy-detection/augmented_resized_V2"
cache_dir = base_dir + '/cache'
train_dir = base_dir + "/train"
test_dir = base_dir + "/test"
val_dir = base_dir + "/val"
g_img_height_pixels = 600
g_img_width_pixels = 600

g_input_channels = 3  # RGB pixels
g_batch_size = 32
g_num_classes = 5
g_training_epochs = 10
latent_dim = 40
AUTOTUNE = tf.data.AUTOTUNE

# Enable mixed precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# Preprocessed training images and combined with preprocessed labels
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    labels="inferred",
    label_mode="int",
    color_mode="rgb",
    class_names=["0", "1", "2", "3", "4"],
    subset=None,
    seed=random.randint(0, 512000),
    image_size=(g_img_height_pixels, g_img_width_pixels),
    batch_size=g_batch_size,
    shuffle=True,
    validation_split=None,
    crop_to_aspect_ratio=False
    # Keep as much info without distortion with the padding caveat
    # pad_to_aspect_ratio=True
)

classes_array = np.arange(g_num_classes)


# @tf.function
def generate_weighed_classes(dataset_, classes_array_):
    targets_labels = np.empty(0, dtype=np.int32, order='C')
    b_ctr = 0
    for _, batches_labels in dataset_:
        # print("batches_labels=", batches_labels, " b_ctr=", b_ctr)
        b_ctr = b_ctr + 1
        # Concat along first dimension '0'
        # targets_labels = tf.concat([targets_labels, batches_labels], 0)
        targets_labels = np.append(targets_labels, batches_labels.numpy())

    ret_class_weights = compute_class_weight(
        class_weight='balanced',
        classes=classes_array_,
        y=targets_labels
    )

    print("targets_labels=", np.shape(targets_labels), " batch_ctr=", b_ctr)
    print("class_weights=", ret_class_weights)
    # Create a dictionary for class weights
    class_weights_dict = {cls: weight for cls, weight in zip(classes_array, ret_class_weights)}
    print(class_weights_dict)

    return class_weights_dict


class_weights = generate_weighed_classes(train_ds, classes_array)

# Preprocessed training images and combined with preprocessed labels
val_ds = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    labels="inferred",
    label_mode="int",
    color_mode="rgb",
    class_names=["0", "1", "2", "3", "4"],
    subset=None,
    seed=random.randint(0, 512000),
    image_size=(g_img_height_pixels, g_img_width_pixels),
    batch_size=g_batch_size,
    shuffle=False,
    validation_split=None,
    crop_to_aspect_ratio=False
    # Keep as much info without distortion with the padding caveat
    # pad_to_aspect_ratio=True
)

max_no_dr_num_allowed = tf.constant(589, dtype=tf.int32)  # Labeled 4 (high disease=708) img samples
# max_no_dr_num_allowed = tf.constant(3, dtype=tf.int32)
allowed_no_dr_samples_counter = tf.Variable(0, dtype=tf.int32)
allowed_dr_samples_counter = tf.Variable(0, dtype=tf.int32)


# Filter executions options when label reports retina does not indicate presence of retinopathy
@tf.function
def filter_by_no_dr_label(max_allowed_no_dr_sample_number):
    global allowed_no_dr_samples_counter
    # tf.print("filter_by_no_dr_label:",
    #           " allowed_no_dr_samples_counter=", allowed_no_dr_samples_counter)
    if tf.less(allowed_no_dr_samples_counter, max_allowed_no_dr_sample_number):
        allowed_no_dr_samples_counter = allowed_no_dr_samples_counter.assign_add(1)
        allow = True
        # tf.print("filter_by_no_dr_label: true=", allow)
    else:
        allow = False
        # tf.print("filter_by_no_dr_label: false=", allow)

    with tf.control_dependencies([allowed_no_dr_samples_counter]):
        return allow


# Filter executions options when label reports retina indicates some presence of retinopathy
@tf.function
def filter_by_dr_label(label):
    global allowed_dr_samples_counter
    allowed_dr_samples_counter = allowed_dr_samples_counter.assign_add(1)
    if tf.equal(label, 4):
        return True
    else:
        return False


@tf.function
def filter_by_label(image, label):
    print("label=", label)
    print("image=", image)
    print("max no dr=", max_no_dr_num_allowed)
    # allowed = (label == 0) & (allowed_no_dr_samples_counter < max_no_dr_num_allowed)
    update_counter = tf.cond(tf.equal(label, 0), lambda: filter_by_no_dr_label(max_no_dr_num_allowed),
                             lambda: filter_by_dr_label(label))
    # print("update_counter", update_counter)
    # include, allowed_no_dr_samples_counter = update_counter
    # tf.print("update_counter", update_counter)
    # return include
    with tf.control_dependencies([update_counter]):
        include = update_counter
        # tf.print("label=", label, " include=", include)
        return include


# Consolidate any substantially perceived level of of desease to just one level for the dataset
def map_to_consolidate_desease_level(img, lbl):
    if lbl > 0:
        lbl = 1

    return img, lbl


# Preprocessed training images and combined with preprocessed labels
test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    labels="inferred",
    label_mode="int",
    color_mode="rgb",
    class_names=["0", "1", "2", "3", "4"],
    subset=None,
    seed=random.randint(0, 512000),
    image_size=(g_img_height_pixels, g_img_width_pixels),
    batch_size=g_batch_size,
    shuffle=False,
    validation_split=None,
    crop_to_aspect_ratio=False
    # Keep as much info without distortion with the padding caveat
    # pad_to_aspect_ratio=True
)

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

train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)

# val_ds = val_ds.take(1).filter(filter_by_label).map(map_to_consolidate_desease_level).batch(1).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# Model with pre-trained base
base_model = tf.keras.applications.ResNet152(input_shape=(g_img_height_pixels, g_img_width_pixels, g_input_channels),
                                             include_top=False
                                             ,
                                             weights='imagenet')

# base_model.trainable = False  # Freeze the base model

inputs = tf.keras.Input(shape=(g_img_height_pixels, g_img_width_pixels, g_input_channels), dtype="uint8")
x = tf.cast(inputs, tf.float32)
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
x = tf.keras.layers.Dropout(0.5)(x)
# outputs = tf.keras.layers.Dense(g_num_classes, activation='softmax')(x)
outputs = tf.keras.layers.Dense(g_num_classes, dtype='float32', activation='softmax')(x)
model = tf.keras.Model(inputs, outputs)


# base_model.trainable = False  # Freeze the base model
# model = tf.keras.Sequential([
#     tf.keras.layers.Rescaling(1. / 255, input_shape=(g_img_height_pixels, g_img_width_pixels, g_input_channels)),
#     data_augmentation,
#     base_model(training=False),
#     tf.keras.layers.Conv2D(
#         filters=64, kernel_size=6, strides=(2, 2), activation='relu', kernel_regularizer=l2(0.0005)),
#     tf.keras.layers.GlobalAveragePooling2D(),
#     tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=l2(0.0005)),
#     tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.Dropout(0.5),
#     # tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=l2(0.0005)),
#     # tf.keras.layers.BatchNormalization(),
#     # tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=l2(0.0005)),
#     # tf.keras.layers.BatchNormalization(),
#     tf.keras.layers.Dense(g_num_classes, activation='softmax')
# ])

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
        tf.print("allowed_no_dr_samples_counter=", allowed_no_dr_samples_counter)


# Callbacks
callbacks = [
    # MyCallback(),
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6),
    # tf.keras.callbacks.ModelCheckpoint('dr_best_class5_model.h5', monitor='val_loss', save_best_only=True)
]

# tuned_learning_rate = 0.001
# model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=tuned_learning_rate),
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(),
#               metrics=['accuracy'])
#
# # model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=tuned_learning_rate),
# #               loss=tf.keras.losses.BinaryCrossentropy(),
# #               metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.5, name='accuracy')])
#
# history = model.fit(
#     train_ds,
#     validation_data=val_ds,
#     epochs=g_training_epochs
#     ,
#     callbacks=callbacks
# )

base_model.trainable = True
# Set approximately the last 25 layers of base model trainable
for layer in base_model.layers[0:500]:
    layer.trainable = False

# base_model.trainable = False


base_model.summary(show_trainable=True)

# tuned_learning_rate = 1e-5 # Low learning rate
tuned_learning_rate = 0.001
model.compile(
    optimizer=tf.keras.optimizers.Adam(tuned_learning_rate),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy'],
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=g_training_epochs,
    class_weight=class_weights,
    callbacks=callbacks
)

del train_ds
del val_ds
gc.collect()

# Evaluate model
model.trainable = False  # Freeze the base model
model.compile(
    optimizer=tf.keras.optimizers.Adam(tuned_learning_rate),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy'],
)
model.summary(show_trainable=True)

print("**************************************")
print("********** Testing Model *************")
print("*************************************")

model.evaluate(
    x=test_ds,
    verbose='auto',
    sample_weight=None,
    steps=None,
    callbacks=None,
    return_dict=False,
)
