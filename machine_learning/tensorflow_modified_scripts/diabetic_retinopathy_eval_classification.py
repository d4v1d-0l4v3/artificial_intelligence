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

# Definitions
# base_dir = "/media/davidolave/OS/Downloads/ai_datasets/diabetic-retinopathy-detection"
base_dir = "/media/davidolave/My Passport/datasets/retina/diabetic-retinopathy-detection_v1"
# labels_file_path = base_dir + "/trainLabels.csv"
test_dir = base_dir + "/test_disease_stages_images"
# Seems (anecdotally determined) that the width to height image ratio is approx 1.5
g_width_to_height_img_ratio = 1.5
# g_img_height_pixels = 1536
g_img_height_pixels = 512
g_img_width_pixels = int (round(g_img_height_pixels * g_width_to_height_img_ratio))
g_eval_model_file_base_name = "best_model_dr_binary_class_2labels"
g_eval_model_file_name_ext = ".h5"
g_eval_model_tflite_name_ext = ".tflite"
g_eval_model_h5_file_path = "./" + g_eval_model_file_base_name + g_eval_model_file_name_ext
g_eval_model_tflite_file_path = "./" + g_eval_model_file_base_name + g_eval_model_tflite_name_ext
g_eval_model_saved_model_path = "./" + g_eval_model_file_base_name + "_save_model/1/"

g_input_channels = 3  # RGB pixels
g_batch_size = 8
g_validation_split = None
g_num_classes = 1
AUTOTUNE = tf.data.AUTOTUNE
# Initially pretrained model name selected for this model
pretrained_model_name = "resnet152"

# Enable mixed precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# Preprocessed training labels file
# df = pd.read_csv(labels_file_path)
# labels_list_to_sort = [(file_name, level) for (file_name, level) in df.itertuples(index=False)]
# sorted_labels_list = sorted(labels_list_to_sort, key=itemgetter(0))  # Sort by filename
# sorted_labels_list = [item[1] for item in sorted_labels_list]
label0_images=25172
sorted_labels_list = np.zeros(label0_images, dtype=int).tolist()
sorted_labels_list[0] = 1

# Preprocessed training images and combined with preprocessed labels
val_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    labels=sorted_labels_list,
    label_mode="binary",
    color_mode="rgb",
    class_names=None,
    subset=None,
    seed=random.randint(0, 512000),
    image_size=(g_img_height_pixels, g_img_width_pixels),
    batch_size=g_batch_size,
    shuffle=False,
    validation_split=g_validation_split,
    crop_to_aspect_ratio=False
    # Keep as much info without distortion with the padding caveat
    # pad_to_aspect_ratio=True
)

# Test delete
# train_ds = train_ds.filter(filter_by_label).map(map_to_consolidate_desease_level).prefetch(buffer_size=AUTOTUNE)
# print("elements in dataset=", train_ds.cardinality(), "unknown=", tf.data.UNKNOWN_CARDINALITY)
#
# # train_ds = train_ds.take(20).filter(filter_by_label)
# # train_imgs = []
# # train_labels = []
# no_dr_ctr = 0
# dr1_ctr = 0
# other_dr_ctr = 0
# for element in train_ds:
#     # print("Processing batch", element)
#     image, label = element
#     # print("label numpy=", label.numpy())
#     if label.numpy() == 0:
#         no_dr_ctr = no_dr_ctr + 1
#     elif label.numpy() == 1:
#         dr1_ctr = dr1_ctr + 1
#     else:
#         other_dr_ctr = other_dr_ctr + 1
#
#
# print("*** End of Filtering images no dr ctr=", no_dr_ctr, " dr1 ctr=", dr1_ctr,
#       " other_dr_ctr=", other_dr_ctr, " dr filter ctr=", allowed_dr_samples_counter)
#
# # Test delete!!!
# quit()

val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

model = tf.keras.models.load_model (g_eval_model_h5_file_path)
model.trainable = False  # Freeze the base model
model.summary(show_trainable=True)

# Get pretrained layer
for layer in model.layers:
    if layer.name == pretrained_model_name:
        pretrain_model = layer
        pretrain_model.trainable = False
        print("pretrain model trainable=", pretrain_model.trainable)

        # pretrain_model.summary(show_trainable=True)

# model.evaluate(
#     x=val_ds,
#     verbose='auto',
#     sample_weight=None,
#     steps=None,
#     callbacks=None,
#     return_dict=False,
# )

# Convert to tflite model
# tf.saved_model.save(model, g_eval_model_saved_model_path)
# converter = tf.lite.TFLiteConverter.from_saved_model(g_eval_model_saved_model_path)
# converter.target_spec.supported_ops = [
#   tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
#   tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
# ]
# tflite_model = converter.convert()
# with open(g_eval_model_tflite_file_path, "wb") as tfliteFileWriter:
#     tfliteFileWriter.write(tflite_model)

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path=g_eval_model_tflite_file_path)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test the model on input data.
tflite_input_shape = input_details[0]['shape']
print ("input shape expected for tflite model:", tflite_input_shape)
val_ds = val_ds.rebatch(1).prefetch(buffer_size=AUTOTUNE)
i = 0
for element in val_ds:
    i = i + 1
    image, label = element
    tflite_input = tf.cast(image, tf.uint8)
    interpreter.set_tensor(input_details[0]['index'], tflite_input)

    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])
    if output_data > 0.5:
        print ("Error: unexpected positive output=", output_data, " sampled=", i)
    # print("inferred:", output_data, "labeled", label)
