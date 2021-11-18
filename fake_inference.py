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





imgs_path = './example_images/'
imgs = os.listdir(imgs_path)
args = Args()

# initialize net   
model = Yolov3Tiny(args)

while True:
    img_file = random.choice(imgs)
    img = cv.imread(imgs_path+img_file)
    img_with_annot, inference_time, predictions = model.predict_img(img, plot=False)

    
    

    url = "localhost:8000/image_request"
    file_obj = {'image': open(imgs_path+img_file, 'rb')}
    payload = {
        'remark1': 'hey',
        'remark2': 'hello',
    }
    r = requests.post("http://127.0.0.1:8000/process_image", files=file_obj, data=payload)

    print(r.status_code)
            
    time.sleep(10000)


            

    