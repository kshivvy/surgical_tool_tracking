import os #os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import re
import shutil
import argparse
import math
import numpy as np
import random
from tqdm import tqdm


ATLAS_DIR = os.path.join(os.path.abspath(os.getcwd()), '..', '..', 'ATLAS_Dione_ObjectDetection', 'ATLAS_Dione_ObjectDetection')
ATLAS_IMAGE_DIR = os.path.join(ATLAS_DIR, 'JPEGImages')
ATLAS_ANNOTATIONS_DIR = os.path.join(ATLAS_DIR, 'Annotations')

M2CAI16_DIR = os.path.join(os.path.abspath(os.getcwd()), '..', '..', 'm2cai16-tool-locations')
M2CAI16_IMAGE_DIR = os.path.join(M2CAI16_DIR, 'JPEGImages')
M2CAI16_ANNOTATIONS_DIR = os.path.join(M2CAI16_DIR, 'Annotations')

OUTPUT_DIR = 'C:\\Users\\kesha\\Documents\\CS499\\surgical_tool_tracking\\workspace\\images'

# Adapted from: https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html

def iterate_dir(source, annotation_dir, dest, ratio, copy_xml):
    source = source.replace('\\', '/')
    dest = dest.replace('\\', '/')
    train_dir = os.path.join(dest, 'train')
    test_dir = os.path.join(dest, 'test')

    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    images = [f for f in os.listdir(source)
              if re.search(r'([a-zA-Z0-9\s_\\.\-\(\):])+(.jpg|.jpeg|.png)$', f)]

    num_images = len(images)
    num_test_images = math.ceil(ratio*num_images)

    for i in range(num_test_images):
        idx = random.randint(0, len(images)-1)
        filename = images[idx]
        copyfile(os.path.join(source, filename),
                 os.path.join(test_dir, filename))
        if copy_xml:
            xml_filename = os.path.splitext(filename)[0]+'.xml'
            print(xml_filename)
            copyfile(os.path.join(annotation_dir, xml_filename),
                     os.path.join(test_dir,xml_filename))
        images.remove(images[idx])

    for filename in images:
        copyfile(os.path.join(source, filename),
                 os.path.join(train_dir, filename))
        if copy_xml:
            xml_filename = os.path.splitext(filename)[0]+'.xml'
            copyfile(os.path.join(source, xml_filename),
                     os.path.join(train_dir, xml_filename))

def split_xmls_and_images(xmls, images, idxs):
    split_xmls =[]
    split_images = []

    for idx in idxs:
        xml = xmls[idx]
        prefix = xml.split('.')[0]
        split_xmls.append(xml)

        for image in images:
            if prefix in image and 'flip' not in image:
                split_images.append(image)

    return split_xmls, split_images

def copy_xmls_and_images(image_dir, annotation_dir, output_dir, xmls, images):
    for filename in tqdm(images):
        shutil.copyfile(os.path.join(image_dir, filename),
                 os.path.join(output_dir, filename))

    for xml in tqdm(xmls):
        shutil.copyfile(os.path.join(annotation_dir, xml),
                 os.path.join(output_dir, xml))

def generate_train_test_split(image_dir, annotation_dir, output_dir, ratio):
    xmls = [xml_file for xml_file in os.listdir(annotation_dir)]
    images = [image for image in os.listdir(image_dir)]

    num_images = len(xmls)
    num_train_images = num_images - math.ceil(ratio * num_images)

    idxs = np.random.permutation(num_images)
    train_idxs, test_idxs = idxs[:num_train_images], idxs[num_train_images:]

    train_xmls, train_images = split_xmls_and_images(xmls, images, train_idxs)
    test_xmls, test_images = split_xmls_and_images(xmls, images, test_idxs)

    OUTPUT_TRAIN_DIR = os.path.abspath(os.path.join(OUTPUT_DIR, 'train'))
    OUTPUT_TEST_DIR = os.path.abspath(os.path.join(OUTPUT_DIR, 'test'))
    copy_xmls_and_images(image_dir, annotation_dir, OUTPUT_TRAIN_DIR, train_xmls, train_images)
    copy_xmls_and_images(image_dir, annotation_dir, OUTPUT_TEST_DIR, test_xmls, test_images)

def main():
    generate_train_test_split(image_dir=M2CAI16_IMAGE_DIR,
                              annotation_dir=M2CAI16_ANNOTATIONS_DIR,
                              output_dir=OUTPUT_DIR,
                              ratio=0.2)

if __name__ == '__main__':
    main()
