
from re import X
import sys
import json
from PIL import Image
import time
import os 
import numpy as np
import cv2
# process annotations from json file, write 
# as txt file with same name as img
def process_bbox(json_file, out_dir):
    with open(str(json_file)) as annotation:
        # Load items in json
        data = json.load(annotation)
        annots = data["annotations"]
        imgs = data['images']

        widths = {}
        heights = {}

        # store img width, height
        for img in imgs:
            widths[img['id']] = img['width']
            heights[img['id']] = img['height']


        for annot in annots:
            id = annot["image_id"]
            x,y,w,h= annot["bbox"]
            cat = annot['category_id']
            xc, yc = x+w/2, y+h/2

            imw, imh = widths[id], heights[id]



            bbox_norm = [xc / imw, yc / imh, w / imw, h / imh]

            towrite = [cat] + bbox_norm
            out_file = os.path.join(out_dir, f'{id}.txt')          
            f = open(out_file, "w+")
            f.write(' '.join([str(i) for i in towrite]))
            f.close()

# displays bounding box over image 
# 0/0---column--->
#  |
#  |
# row
#  |
#  |
#  v
# coco format json annotations: x, y, width, height 
def overlay_bbox(dir, id):
    img_file = os.path.join(dir, f'{id}.jpg')
    bbox_file  = os.path.join(dir, f'{id}.txt')  
    x1,y1,x2,y2= np.loadtxt(bbox_file)
    img =  cv2.imread(img_file)
    

    img_with_bbox = cv2.rectangle(img, (int(x1),int(y1)), (int(x2),int(y2)), (0,255,0), 2)
    cv2.imshow('img_with_bbox', img_with_bbox)
    cv2.waitKey()

# cd darknet
json_file = 'custom_cfg/wildlife.json'
out_dir = 'custom_dataset/'
process_bbox(json_file, out_dir)

# overlay_bbox(out_dir, 59)

