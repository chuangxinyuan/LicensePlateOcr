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
import argparse
from LPRNet import build_lprnet

from PIL import Image,ImageDraw,ImageFont

import numpy as np

import cv2
import os

import ast
import torch



CHARS = ['京', '沪', '津', '渝', '冀', '晋', '蒙', '辽', '吉', '黑',
         '苏', '浙', '皖', '闽', '赣', '鲁', '豫', '鄂', '湘', '粤',
         '桂', '琼', '川', '贵', '云', '藏', '陕', '甘', '青', '宁',
         '新',
         '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
         'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K',
         'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
         'W', 'X', 'Y', 'Z', 'I', 'O', '-'
         ]

CHARS_DICT = {i:char for i, char in enumerate(CHARS)}





def create_model(args):
    lprnet = build_lprnet(lpr_max_len=args.lpr_max_len, phase=args.phase_train, class_num=len(CHARS), dropout_rate=args.dropout_rate)
    device = torch.device("cuda:0" if args.cuda else "cpu")
    lprnet.to(device)
    print("Successful to build network!")

    # load pretrained model
    if args.pretrained_model:
        lprnet.load_state_dict(torch.load(args.pretrained_model, map_location=torch.device('cpu')))
        print("load pretrained model successful!")
    else:
        print("[Error] Can't found pretrained mode, please check!")
        return False
    return lprnet 

def run(args, annotations):
  model = create_model(args)


  for path,boxes in annotations.items():
      print("Processing: ", path) 
      img = cv2.imread(os.path.join('/mnt/data/datasets/images', os.path.basename(path)))
      for box in boxes:
          img_cropped = img[box['ymin']:box['ymax']+1, box['xmin']:box['xmax']+1]
          height, width, _ = img_cropped.shape
          if height != args.img_size[1] or width != args.img_size[0]:
              img_cropped = cv2.resize(img_cropped, args.img_size)

          img_cropped = img_cropped.astype('float32')
          img_cropped -= 127.5
          img_cropped *= 0.0078125
          img_cropped = np.expand_dims(np.transpose(img_cropped, (2, 0, 1)), axis=0) 


          prebs = model(torch.tensor(img_cropped))
          # greedy decode
          prebs = prebs.cpu().detach().numpy()
          preb_labels = list()
          for i in range(prebs.shape[0]):
              preb = prebs[i, :, :]
              preb_label = list()
              for j in range(preb.shape[1]):
                  preb_label.append(np.argmax(preb[:, j], axis=0))
              no_repeat_blank_label = list()
              pre_c = preb_label[0]
              if pre_c != len(CHARS) - 1:
                  no_repeat_blank_label.append(pre_c)
              for c in preb_label: # dropout repeate label and blank label
                  if (pre_c == c) or (c == len(CHARS) - 1):
                      if c == len(CHARS) - 1:
                          pre_c = c
                      continue
                  no_repeat_blank_label.append(c)
                  pre_c = c
              preb_labels.append(no_repeat_blank_label)

          output = ''.join([str(CHARS_DICT[i.item()]) for i in preb_labels[0]])


          cv2.rectangle(img,(box['xmin'],box['ymin']),(box['xmax'],box['ymax']),(0,255,0),3)
          font = ImageFont.truetype('yahei_mono_0.ttf',40)
          img_pil = Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
          draw = ImageDraw.Draw(img_pil)
          draw.text((box['xmin'],box['ymax']+40),output.replace('?',''),font=font,fill=(0,255,0)) 
          img_ocv = np.array(img_pil)                    
          img = cv2.cvtColor(img_ocv,cv2.COLOR_RGB2BGR)
#             file_writer.write([pr_bytes.decode('utf-8') for pr_bytes in predictions.tolist()][0])
      cv2.imwrite('/mnt/output/'+os.path.basename(path), img)
  

def get_parser():
    parser = argparse.ArgumentParser(description='parameters to train net')
    parser.add_argument('--license_boxes_json_path', default='/mnt/data/outputdata/output.json', help='detection output ')
    parser.add_argument('--batch_size', default=1, help='the batch size')

    parser.add_argument('--img_size', default=(94, 24), help='the image size')
    parser.add_argument('--dropout_rate', default=0, help='dropout rate.')
    parser.add_argument('--lpr_max_len', default=8, help='license plate number max length.')
    parser.add_argument('--phase_train', default=False, type=bool, help='train or test phase flag.')
    parser.add_argument('--cuda', default=False, type=bool, help='Use cuda to train model')
    parser.add_argument('--pretrained_model', default='./lpr_model.pth', help='pretrained base model')

    args = parser.parse_args()

    return args

def main():
  args = get_parser()
  data = open(args.license_boxes_json_path, 'r')
  annotations = ast.literal_eval(data.read())['annotations']

  if not os.path.exists('/mnt/output/'):
      os.makedirs('/mnt/output/')

  run(args, annotations)


if __name__ == '__main__':
  main()
