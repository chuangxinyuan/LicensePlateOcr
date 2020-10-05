# Script to generate cropped images based on bounding boxes for OCR.

from PIL import Image
import xml.etree.ElementTree as ET
import argparse
import os

def get_crops(args):
    tree = ET.parse(args.xml_file)
    root = tree.getroot()
    count = 0

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    for image in root.iter('image'):
       
        img = Image.open(os.path.join(args.image_dir, image.attrib['name']))
        for idx, box in enumerate(image.findall('box')):
            cropped = img.crop((float(box.attrib['xtl']), float(box.attrib['ytl']), float(box.attrib['xbr']), float(box.attrib['ybr'])))
            cropped.save(os.path.join(args.output_dir, image.attrib['name'][:-4]+'_'+str(idx)+image.attrib['name'][-4:]))
            _ = open(os.path.join(args.output_dir, image.attrib['name'][:-4]+'_'+str(idx)+'.txt'), 'w') #empty txt file to be filled with annotations
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--xml_file', default='./license-detector.xml')
    parser.add_argument('--image_dir', default='./final/')
    parser.add_argument('--output_dir', default='./cropped/')
    args = parser.parse_args()
    get_crops(args)
