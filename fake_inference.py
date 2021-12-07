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





imgs_path = './darknet/small_dataset/'
imgs = os.listdir(imgs_path)

while True:
    img_file = random.choice(imgs)

    if img_file.endswith('txt'):continue
    img = cv.imread(imgs_path+img_file)

    # sends an image every 5 seconds
    try:
        file_obj = {'image': open(imgs_path+img_file, 'rb')}
        r = requests.post("http://127.0.0.1:8000/process_image", files=file_obj)
        # min number seconds before next image sent 
        time.sleep(5)

    except Exception as e:
        print(str(e))
   
    
    

    


            

    