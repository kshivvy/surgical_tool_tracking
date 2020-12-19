import os
import glob
import pandas as pd
import io
import xml.etree.ElementTree as ET
import argparse
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
import tensorflow.compat.v1 as tf
from PIL import Image
from object_detection.utils import dataset_util, label_map_util
from collections import namedtuple

ATLAS_DIR = os.path.join(os.path.abspath(os.getcwd()), '..', '..', 'ATLAS_Dione_ObjectDetection', 'ATLAS_Dione_ObjectDetection')
ATLAS_IMAGE_DIR = os.path.join(ATLAS_DIR, 'JPEGImages')
ATLAS_ANNOTATIONS_DIR = os.path.join(ATLAS_DIR, 'Annotations')

M2CAI16_DIR = os.path.join(os.path.abspath(os.getcwd()), '..', '..', 'm2cai16-tool-locations')
M2CAI16_IMAGE_DIR = os.path.join(M2CAI16_DIR, 'JPEGImages')
M2CAI16_ANNOTATIONS_DIR = os.path.join(M2CAI16_DIR, 'Annotations')


OUTPUT_DIR = 'C:\\Users\\kesha\\Documents\\CS499\\surgical_tool_tracking\\workspace\\images'
ANNOTATIONS_DIR = 'C:\\Users\\kesha\\Documents\\CS499\\surgical_tool_tracking\\workspace\\annotations'

LABELS_PATH = os.path.join(ANNOTATIONS_DIR, 'label_map.pbtxt')
label_map_dict = label_map_util.get_label_map_dict(label_map_util.load_labelmap(LABELS_PATH))
column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']

# Adapted from: https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html

def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)

    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df

def class_text_to_int(row_label):
    return label_map_dict[row_label]

def split_df(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]

def create_tf_example(group, path):
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example

def main(_):
    split = 'test'
    tf_record_path = os.path.join(ANNOTATIONS_DIR, split + '.record')
    split_dir = os.path.join(OUTPUT_DIR, split)

    writer = tf.python_io.TFRecordWriter(tf_record_path)
    examples = xml_to_csv(split_dir)
    grouped = split_df(examples, 'filename')

    for group in grouped:
        tf_example = create_tf_example(group, split_dir)
        writer.write(tf_example.SerializeToString())
    writer.close()

    print('Successfully created the TFRecord file: {}'.format(tf_record_path))


if __name__ == '__main__':
    tf.app.run()
