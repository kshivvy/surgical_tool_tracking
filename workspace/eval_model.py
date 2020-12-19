import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
# tf.logging.set_verbosity(tf.logging.INFO)
from object_detection import model_lib_v2

# Set CPU as available physical device
my_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
tf.config.experimental.set_visible_devices(devices= my_devices, device_type='CPU')

# To find out which devices your operations and tensors are assigned to
# tf.debugging.set_log_device_placement(True)

# Source: https://github.com/tensorflow/models/issues/8876
strategy = tf.compat.v2.distribute.OneDeviceStrategy(device="/cpu:0")

with strategy.scope():
    model_lib_v2.eval_continuously(
        pipeline_config_path='models/faster_rcnn_inception_resnet_v2_640x640_coco17/pipeline.config',
        model_dir='models/faster_rcnn_inception_resnet_v2_640x640_coco17',
        checkpoint_dir='models/faster_rcnn_inception_resnet_v2_640x640_coco17',
        checkpoint_every_n=1)
