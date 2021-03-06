
from re import X
import sys
import json
from PIL import Image
import time
import os 
import numpy as np
import cv2
import shutil


    


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
            # top left corner, width, height 
            x,y,w,h= annot["bbox"]
            cat = annot['category_id']
                
            xc, yc = x+w/2, y+h/2

            imw, imh = widths[id], heights[id]



            bbox_norm = [xc / imw, yc / imh, w / imw, h / imh]

            towrite = [cat] + bbox_norm
            out_file = os.path.join(out_dir, f'{id}.txt')          
            f = open(out_file, "w")

            # if id == "6995":
            #     print('out_file', out_file)
            #     print(towrite)
            #     exit(-1)
            # print('writing file', id)
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
    label, xc,yc,w,h= np.loadtxt(bbox_file)
    
    print('label:', label)
    img =  cv2.imread(img_file)
    

    im_w, im_h, _ = img.shape
    x1 = (xc-(h/2))*im_h
    x2 = (xc+(h/2))*im_h
    y1 = (yc-(w/2))*im_w
    y2 = (yc+(w/2))*im_w
    print('BBOX', x1,y1,x2,y2)
    

    img_with_bbox = cv2.rectangle(img, (int(x1),int(y1)), (int(x2),int(y2)), (0,255,0), 2)
    cv2.imshow('img_with_bbox', img_with_bbox)
    cv2.waitKey()

# cd darknet
json_file = 'custom_cfg/wildlife.json'
out_dir = 'custom_dataset/'
process_bbox(json_file, out_dir)

# 530->690
# 1087+1208
# overlay_bbox(out_dir, 1671)

