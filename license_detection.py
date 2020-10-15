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


def generateXML(filename,outputPath,w,h,d,boxes):
    top = ET.Element('annotation')
    childFolder = ET.SubElement(top, 'folder')
    childFolder.text = 'images'
    childFilename = ET.SubElement(top, 'filename')
    childFilename.text = filename[0:filename.rfind(".")]
    childPath = ET.SubElement(top, 'path')
    childPath.text = outputPath + "/" + filename
    childSource = ET.SubElement(top, 'source')
    childDatabase = ET.SubElement(childSource, 'database')
    childDatabase.text = 'Unknown'
    childSize = ET.SubElement(top, 'size')
    childWidth = ET.SubElement(childSize, 'width')
    childWidth.text = str(w)
    childHeight = ET.SubElement(childSize, 'height')
    childHeight.text = str(h)
    childDepth = ET.SubElement(childSize, 'depth')
    childDepth.text = str(d)
    childSegmented = ET.SubElement(top, 'segmented')
    childSegmented.text = str(0)
    #boxes tiene que contener labels
    for (box,score) in boxes:
        # Cambiar categoria por label
        category = box[0]
        box = box[1].astype("int")
        ####### 
        # Cuidado esto está cambiado con respecto a lo que es habitualmente
        #######  
        (x,y,xmax,ymax) = box
        childObject = ET.SubElement(top, 'object')
        childName = ET.SubElement(childObject, 'name')
        childName.text = category
        childScore = ET.SubElement(childObject, 'confidence')
        childScore.text = str(score)
        childPose = ET.SubElement(childObject, 'pose')
        childPose.text = 'Unspecified'
        childTruncated = ET.SubElement(childObject, 'truncated')
        childTruncated.text = '0'
        childDifficult = ET.SubElement(childObject, 'difficult')
        childDifficult.text = '0'
        childBndBox = ET.SubElement(childObject, 'bndbox')
        childXmin = ET.SubElement(childBndBox, 'xmin')
        childXmin.text = str(x)
        childYmin = ET.SubElement(childBndBox, 'ymin')
        childYmin.text = str(y)
        childXmax = ET.SubElement(childBndBox, 'xmax')
        childXmax.text = str(xmax)
        childYmax = ET.SubElement(childBndBox, 'ymax')
        childYmax.text = str(ymax)
    return prettify(top)

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
                result.append(([label, [xmin, ymin, xmax, ymax]], scores[0][i]))
        output['annotations'][imagePath]  = str(result)
        # parse the filename from the input image path, construct the
        # path to the output image, and write the image to disk
#         filename = imagePath.split(os.path.sep)[-1]
#         #outputPath = os.path.sep.join([args["output"], filename])
#         file = open(imagePath[0:imagePath.rfind(".")]+".xml", "w")
#         file.write(generateXML(imagePath[0:imagePath.rfind(".")],imagePath,width, height, 3, result))
#         file.close()
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
