import tensorflow as tf
import os
import numpy as np
import logging
import io
import PIL.Image
import hashlib
import random

from object_detection.utils import dataset_util
import cv2 as cv
from utils.utils import read_config
from object_detection.dataset_tools import tf_record_creation_util
from object_detection.utils import label_map_util
import hashlib

config = read_config()
save_dir = config['TFRecord']['save_dir']
IMAGE_DIR = config['TFRecord']['IMAGE_DIR']
TF_REC_DIR = config['TFRecord']['TF_REC_DIR']
data_dir = config['TFRecord']['data_dir']

flags = tf.app.flags
flags.DEFINE_string('output_dir', TF_REC_DIR, 'Path to output TFRecord')
FLAGS = flags.FLAGS


def create_tf_example(data):
  img_path = os.path.join(IMAGE_DIR, data['filename'])
  with tf.gfile.GFile(img_path, 'rb') as fid:
    encoded_jpg = fid.read()
  encoded_jpg_io = io.BytesIO(encoded_jpg)
  
  image = PIL.Image.open(encoded_jpg_io)
  if image.format != 'JPEG':
    raise ValueError('Image format not JPEG')
  
  key = hashlib.sha256(encoded_jpg).hexdigest()

  width = int(data['size']['width'])
  height = int(data['size']['height'])

  xmins = []
  ymins = []
  xmaxs = []
  ymaxs = []
  classes = []
  classes_text = []
 
  if 'object' in data:
    for obj in data['object']:
      xmins.append(obj[0] / width)
      ymins.append(obj[1] / height)
      xmaxs.append(obj[2] / width)
      ymaxs.append(obj[3] / height)
      class_name = 'PCB'
      classes_text.append(class_name.encode('utf8'))
      classes.append(1)

  feature_dict = {
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(
          data['filename'].encode('utf8')),
      'image/source_id': dataset_util.bytes_feature(
          data['filename'].encode('utf8')),
      'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
      'image/encoded': dataset_util.bytes_feature(encoded_jpg),
      'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
  }
  example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
  return example


def main()->None:
    #writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
  if not(os.path.exists(IMAGE_DIR)):
    os.mkdir(IMAGE_DIR)
  if not(os.path.exists(TF_REC_DIR)):
    os.mkdir(TF_REC_DIR)

  # TODO(user): Write code to read in your dataset to examples variable
  count=0
  examples = []
  for file in os.listdir(save_dir):
    example = {}
    file_path = (os.path.join(save_dir, file))
    data = np.load(file_path,allow_pickle=True)
    # print(data.keys())
    image = np.array(data['image'],dtype=np.float32)
    filename = (os.path.join(str(count)+'.jpg'))
    cv.imwrite(os.path.join(IMAGE_DIR,str(count)+'.jpg'),image*255)
    count+=1 
    example['filename'] = filename
    example['size'] = {}
    example['size']['width'] = 512
    example['size']['height'] = 512
    example['object'] = data['boxes']
    examples.append(example)
      # tf_example = create_tf_example(example)
      # writer.write(tf_example.SerializeToString())

      # writer.close()
  image_dir = os.path.join(data_dir, 'images')
  annotations_dir = os.path.join(data_dir, 'annotations')
  examples_list = list(range(1,len(examples)))
  random.shuffle(examples_list)

  # Test images are not included in the downloaded data set, so we shall perform
  # our own split.
 
  num_examples = len(examples_list)
  num_train = int(0.9 * num_examples)
  train_examples = examples_list[:num_train]
  val_examples = examples_list[num_train:]
  logging.info('%d training and %d validation examples.',
               len(train_examples), len(val_examples))

  train_output_path = os.path.join(FLAGS.output_dir, 'pcb_train.record')
  val_output_path = os.path.join(FLAGS.output_dir, 'pcb_val.record')

  writer = tf.python_io.TFRecordWriter(train_output_path)
  for idx in train_examples:
      example = examples[idx]
      tf_example = create_tf_example(example)
      writer.write(tf_example.SerializeToString())

  writer.close()

  writer = tf.python_io.TFRecordWriter(val_output_path)
  for idx in val_examples:
      example = examples[idx]
      tf_example = create_tf_example(example)
      writer.write(tf_example.SerializeToString())





if __name__ == '__main__':
  tf.app.run()