# This code is written at BigVision LLC. It is based on the OpenCV project. It is subject to the license terms in the LICENSE file found in this distribution and at http://opencv.org/license.html

# Real-time object detection with YoloV3
import cv2 as cv
import argparse
import sys
import numpy as np
from urllib.request import urlopen
import os
import datetime
import time
import random
from inference import Yolov3Tiny



class Args:
    def __init__(self, conf_threshold=0.25,
                       nms_threshold=0.4,
                       img_width=416,
                       img_height=416,
                       cfg_path='./darknet/cfg/',
                       data_path='./darknet/data/',
                       model='yolov3-tiny',
                       classes_file='coco.names',
                       webserver_url='http://192.168.12.233:81/stream'):
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.img_width = img_width
        self.img_height = img_height
        self.cfg_path = cfg_path
        self.data_path = data_path
        self.model = model
        self.classes_file = classes_file
        self.webserver_url=webserver_url


imgs_path = './example_images/'
imgs = os.listdir(imgs_path)
args = Args()

# initialize net   
model = Yolov3Tiny(args)

while True:
    img_file = random.choice(imgs)
    img = cv.imread(imgs_path+img_file)


    print('processing img', img)

    model.predict_img(img, plot=True)

    # time.sleep(2)


            

    