# USAGE
# python predict_batch.py --input logos/images --output output

# import the necessary packages
import numpy as np
import xml.etree.ElementTree as ET
from xml.dom import minidom
import numpy as np
import argparse
import cv2
import os
from PIL import Image
import glob
import tensorflow as tf
import json


def prettify(elem):
    """Return a pretty-printed XML string for the Element.
    """
    rough_string = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")

def _normalize_box(box, w, h):
    xmin = int(box[1] * w)
    ymin = int(box[0] * h)
    xmax = int(box[3] * w)
    ymax = int(box[2] * h)
    return xmin, ymin, xmax, ymax

def load_image_into_numpy(image):
	(im_width, im_height) = image.size
	return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

# loop over the input image paths

def mainDataset(dataset,weights,output_path):
    CLASSES = {1:'license', 2:'license'}
   
    
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(weights , 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
            config = tf.ConfigProto()
            config.gpu_options.allow_growth=True
            sess = tf.Session(graph=detection_graph, config=config)

    imagePaths = glob.glob(dataset+"*.jpg") + glob.glob(dataset+"*.png") + glob.glob(dataset+"*.jpeg")
    
    output = {'annotations':{}}
    for (i, imagePath) in enumerate(imagePaths):
        # load the input image (in BGR order), clone it, and preprocess it
        #print("[INFO] predicting on image {} of {}".format(i + 1,
        #	len(imagePaths)))
        print("Processing: ", imagePath)
        # load the input image (in BGR order), clone it, and preprocess it
        image = cv2.imread(imagePath)
        width, height = image.shape[:2]
        image_pil = Image.fromarray(image)
        if width > 1920 or height > 1080:
            image_pil = image_pil.resize((width // 2, height // 2), Image.ANTIALIAS)
        image_np = np.array(image_pil)
        image_np_expanded = np.expand_dims(image_np, axis=0)

        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        scores = detection_graph.get_tensor_by_name('detection_scores:0')
        classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        (boxes, scores, classes, num_detections) = sess.run([boxes, scores, classes, num_detections], feed_dict={image_tensor: image_np_expanded})

        result = []
        for i in range(len(classes[0])):
            if scores[0][i] >= 0.5:
                xmin, ymin, xmax, ymax = _normalize_box(boxes[0][i], width, height)
                label = CLASSES[classes[0][i]]
                result.append({'label':label, 'xmin': xmin, 'ymin':ymin, 'xmax':xmax, 'ymax':ymax})
        output['annotations'][imagePath]  = str(result)
      
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    json.dump(output, open(output_path+'output.json', 'w'))
    print(output)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='/data/test/')
    parser.add_argument('--weights', default='/data/frozen_inference_graph.pb')
    parser.add_argument("--output_path", default="/mnt/output/")
    args = parser.parse_args()
    mainDataset(args.dataset, args.weights, args.output_path)
