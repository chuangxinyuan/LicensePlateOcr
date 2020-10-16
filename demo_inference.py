"""A script to run inference on a set of image files.

NOTE #1: The Attention OCR model was trained only using FSNS train dataset and
it will work only for images which look more or less similar to french street
names. In order to apply it to images from a different distribution you need
to retrain (or at least fine-tune) it using images from that distribution.

NOTE #2: This script exists for demo purposes only. It is highly recommended
to use tools and mechanisms provided by the TensorFlow Serving system to run
inference on TensorFlow models in production:
https://www.tensorflow.org/serving/serving_basic

Usage:
python demo_inference.py --batch_size=32 \
  --checkpoint=model.ckpt-399731\
  --image_path_pattern=./datasets/data/fsns/temp/fsns_train_%02d.png
"""
import numpy as np
import PIL.Image
import tensorflow as tf
from tensorflow.python.platform import flags
from tensorflow.python.training import monitored_session
import cv2
import os
import common_flags
import datasets
import ast
import data_provider
import glob
FLAGS = flags.FLAGS
common_flags.define()

# e.g. ./datasets/data/fsns/temp/fsns_train_%02d.png
flags.DEFINE_string('image_path_pattern', '',
                    'A file pattern with a placeholder for the image index.')
flags.DEFINE_string('license_boxes_json_path', '/data/output.json', 'A json file with coordinates of bounding boxes for license plates')


def get_dataset_image_size(dataset_name):
  # Ideally this info should be exposed through the dataset interface itself.
  # But currently it is not available by other means.
  ds_module = getattr(datasets, dataset_name)
  height, width, _ = ds_module.DEFAULT_CONFIG['image_shape']
  return width, height


def load_images(batch_size, dataset_name, annotations):
  width, height = get_dataset_image_size(dataset_name)
  images_actual_data = np.ndarray(shape=(len(annotations.values()), height, width, 3),
                                  dtype='uint8')
  count = 0
  for path,boxes in annotations.items():
        print("Processing: ", path) 
        img = cv2.imread(os.path.join('/mnt/data/datasets/images', os.path.basename(path)))
        for box in boxes:
            print(box)
            print(img.shape)
            img_cropped = img[box['xmin']:box['xmax']+1, box['ymin']:box['ymax']+1]
            cv2.imwrite('/data/{}.jpg'.format(count), img_cropped)
            pil_img = PIL.Image.fromarray(img_cropped)
            img = pil_img.resize((width, height), PIL.Image.ANTIALIAS)
            images_actual_data[count, ...] = np.asarray(img)
            count += 1
  return images_actual_data


def create_model(batch_size, dataset_name):
  width, height = get_dataset_image_size(dataset_name)
  dataset = common_flags.create_dataset(split_name=FLAGS.split_name)
  model = common_flags.create_model(
      num_char_classes=dataset.num_char_classes,
      seq_length=dataset.max_sequence_length,
      num_views=dataset.num_of_views,
      null_code=dataset.null_code,
      charset=dataset.charset)
  raw_images = tf.compat.v1.placeholder(
      tf.uint8, shape=[batch_size, height, width, 3])
  images = tf.map_fn(data_provider.preprocess_image, raw_images,
                     dtype=tf.float32)
  endpoints = model.create_base(images, labels_one_hot=None)
  return raw_images, endpoints


def run(checkpoint, batch_size, dataset_name, image_path_pattern, annotations):
  images_placeholder, endpoints = create_model(batch_size,
                                               dataset_name)
  session_creator = monitored_session.ChiefSessionCreator(
      checkpoint_filename_with_path=checkpoint)
  count = 0
  width, height = get_dataset_image_size(dataset_name)
  with monitored_session.MonitoredSession(
          session_creator=session_creator) as sess:
    for path,boxes in annotations.items():
        print("Processing: ", path) 
        img = cv2.imread(os.path.join('/mnt/data/datasets/images', os.path.basename(path)))
        for box in boxes:
            img_cropped = img[box['xmin']:box['xmax']+1, box['ymin']:box['ymax']+1]
            pil_img = PIL.Image.fromarray(img_cropped)
            img = pil_img.resize((width, height), PIL.Image.ANTIALIAS)
            count += 1
            predictions = sess.run(endpoints.predicted_text,
                           feed_dict={images_placeholder: np.asarray(img)[np.newaxis, ...]})
            file_writer = open('/mnt/output/'+os.path.basename(path).split('.')[0]+'.txt', 'w')
            file_writer.write([pr_bytes.decode('utf-8') for pr_bytes in predictions.tolist()][0])
  


def main(_):
  data = open(FLAGS.license_boxes_json_path, 'r')
  annotations = ast.literal_eval(data.read())['annotations']
  FLAGS.batch_size = 1
  if not os.path.exists('/mnt/output/'):
      os.makedirs('/mnt/output/')
  d = open(os.path.join(FLAGS.checkpoint, 'checkpoint'), 'r')
  for line in d.readlines():
    if not 'all_model_checkpoint_path' in line:
        checkpoint = os.path.basename(line.split(':')[1].strip().strip('"'))
        break
  run(os.path.join(FLAGS.checkpoint, checkpoint), FLAGS.batch_size, FLAGS.dataset_name,
                    FLAGS.image_path_pattern, annotations)


if __name__ == '__main__':
  tf.compat.v1.app.run()
