# Script to generate TFRECORDS for attention-ocr.


from random import shuffle
import numpy as np
import glob
# disable TF2 
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
import cv2
import sys
import os
import argparse
from PIL import Image


def encode_utf8_string(text, length, dic, null_id):
    char_ids_padded = [null_id]*length
    char_ids_unpadded = [null_id]*len(text)
    for i in range(len(text)):
        #if i == '\n':
         #   continue
        hash_id = dic[text[i]]
        char_ids_padded[i] = hash_id
        char_ids_unpadded[i] = hash_id
    return char_ids_padded, char_ids_unpadded

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def get_tfrecords(args):
    dict={}
    with open(args.charset_path, encoding="utf") as dict_file:
        for line in dict_file:
            if len(line.strip()) < 2:
                continue # skip errorneous lines
            (key, value) = line.strip().split('\t')
            dict[value] = int(key)
    dict.update({' ':0})

    image_paths = glob.glob(os.path.join(args.data_dir, '*.jpg'))
    image_paths.extend(glob.glob(os.path.join(args.data_dir, '*.png')))

    label_paths = glob.glob(os.path.join(args.data_dir, '*.txt'))

    tfrecord_writer  = tf.python_io.TFRecordWriter(args.output_tfrecord) 
    for j in range(len(image_paths)):
        print(j)
        img = Image.open(image_paths[j])
        orig_width = img.size[0]
        img = img.resize((args.resize_width, args.resize_height), Image.ANTIALIAS)
        np_data = np.array(img)
        print(image_paths[j])
        for text in open(label_paths[j], encoding="utf"):
            print(text)
            char_ids_padded, char_ids_unpadded = encode_utf8_string(
                                text=text.strip('\n'),
                                dic=dict,
                                length=args.text_length,
                                null_id=args.null_id)
            print("done")

        example = tf.train.Example(features=tf.train.Features(
                            feature={
                                'image/encoded': _bytes_feature(np_data.tostring()),
                                'image/format': _bytes_feature(b"raw"),
                                'image/width': _int64_feature([np_data.shape[1]]),
                                'image/orig_width': _int64_feature([orig_width]),
                                'image/class': _int64_feature(char_ids_padded),
                                'image/unpadded_class': _int64_feature(char_ids_unpadded),
                                'image/text': _bytes_feature(bytes(text, 'utf-8')),
                                'image/height': _int64_feature([np_data.shape[0]]),
                            }
                        ))
        tfrecord_writer.write(example.SerializeToString())
    tfrecord_writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--charset_path', default='./charset_size.txt')
    parser.add_argument('--data_dir', default='./data/')
    parser.add_argument('--output_dir', default='./cropped/')
    parser.add_argument('--output_tfrecord', default='./tfrecord_train')
    parser.add_argument('--resize_width', default=300, type=int)
    parser.add_argument('--resize_height', default=80, type=int)
    parser.add_argument('--text_length', default=37, type=int)
    parser.add_argument('--null_id', default=133, type=int)
    args = parser.parse_args()
    get_tfrecords(args)
