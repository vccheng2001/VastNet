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
from inference import Yolov3Tiny, Args
import requests
import random



root_dir = './darknet'
test_data = os.path.join(root_dir, 'small_cfg/test.txt')
imgs = []
with open(test_data) as file:
    for f in file:
        imgs.append(os.path.join(root_dir, f.strip()))

while True:
    img_file = random.choice(imgs)

    if img_file.endswith('txt'):continue
    img = cv.imread(img_file)

    # sends an image every 5 seconds
    try:
        file_obj = {'image': open(img_file, 'rb')}
        r = requests.post("http://127.0.0.1:8000/process_image", files=file_obj)
        # number seconds before next image sent 
        sleep_time =  random.randrange(3,5)
        time.sleep(sleep_time)

    except Exception as e:
        print(str(e))
   
    
    

    


            

    