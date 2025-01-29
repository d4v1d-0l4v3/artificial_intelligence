import tensorflow as tf
from tensorflow.python.client import device_lib
import tensorrt

print(device_lib.list_local_devices())
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
print("Tensorrt version=", tensorrt.__version__)
print("Tensorflow version=", tf.__version__)

