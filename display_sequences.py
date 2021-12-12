
import os
import json
import glob
import shutil
import numpy as np
import cv2 as cv
import requests
import random
import time

def disp():
    root_dir = './darknet'
    imgs = os.listdir(os.path.join(root_dir, 'small_dataset'))
    imgs = reversed(sorted(imgs))
    i = 0
    for im in imgs:
        if not im.endswith('jpg'): continue
        i += 1
        im = os.path.join(root_dir, 'small_dataset', im)
        frame = cv.imread(im)
        cv.imshow('Window', frame)
        key = cv.waitKey(250) # pauses before fetching next image
        

        if i % 10000 == 0: 
            # sends an image every n seconds
            try:
                file_obj = {'image': open(im, 'rb')}
                r = requests.post("http://127.0.0.1:8000/process_image", files=file_obj)
                # sleep_time =  random.randrange(2,4)
                # time.sleep(sleep_time)

            except Exception as e:
                print(str(e))

        if key == 27:#if ESC is pressed, exit loop
            cv.destroyAllWindows()
            break 
disp()