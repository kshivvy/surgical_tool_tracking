import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
# tf.logging.set_verbosity(tf.logging.INFO)
from object_detection import model_lib_v2

# Set CPU as available physical device
my_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
tf.config.experimental.set_visible_devices(devices= my_devices, device_type='CPU')

# To find out which devices your operations and tensors are assigned to
# tf.debugging.set_log_device_placement(True)

num_workers = 1
if num_workers > 1: # DOES NOT WORK!
    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
else:
    strategy = tf.compat.v2.distribute.MirroredStrategy()

with strategy.scope():
    model_lib_v2.train_loop(
        pipeline_config_path='models/faster_rcnn_inception_resnet_v2_640x640_coco17/pipeline.config',
        model_dir='models/faster_rcnn_inception_resnet_v2_640x640_coco17',
        train_steps=1000,
        use_tpu=False,
        checkpoint_every_n=1,
        record_summaries=True)
