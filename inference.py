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
import matplotlib
import matplotlib.pyplot as plt


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


class Yolov3Tiny:
    def __init__(self, args):


        self.conf_threshold = args.conf_threshold
        self.nms_threshold = args.nms_threshold
        self.img_width = args.img_width
        self.img_height = args.img_height
        self.cfg_path = args.cfg_path
        self.data_path = args.data_path
        self.model = args.model
        self.classes_file = f"{self.data_path}{args.classes_file}"
        self.webserver_url=args.webserver_url

        


        
        with open(self.classes_file, 'rt') as f:
            self.classes = f.read().rstrip('\n').split('\n')

        # Give the configuration and weight files for the model and load the network using them.
        model_cfg = f"{self.cfg_path}{self.model}.cfg"
        model_weights = f"{self.cfg_path}{self.model}.weights"

        # load network using cfg, weights
        self.net = cv.dnn.readNetFromDarknet(model_cfg, model_weights)
        self.net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU) 

        self.num_imgs_processed = 0

  

    def begin_stream(self):

        # Process inputs
        window_name = 'ESP32-CAM Object Detection with Yolo'
        cv.namedWindow(window_name, cv.WINDOW_NORMAL)

        # Webcam input
        CAMERA_BUFFER_SIZE=4096
        stream=urlopen(self.webserver_url)
        bts=b''

        while cv.waitKey(1) < 0:
            chunk = stream.read(CAMERA_BUFFER_SIZE) 
            bts+=chunk
            jpghead=bts.find(b'\xff\xd8')
            jpgend=bts.find(b'\xff\xd9')
            if jpghead>-1 and jpgend>-1:

                jpg=bts[jpghead:jpgend+2]
                bts=bts[jpgend+2:]

                try:
                    img=cv.imdecode(np.frombuffer(jpg,dtype=np.uint8),cv.IMREAD_UNCHANGED)
                    self.predict_img(img)
                except:
                    continue


                

    def predict_img(self, img, plot=False):

        self.num_imgs_processed += 1
    
        # blobFromImage performs mean subtraction, scaling, channel swapping
        # creates 4D blob from image
        # resizes, crops image from center
        # subtracts mean values, scales by scalefactor
        # swaps blue, red channels

 
        blob = cv.dnn.blobFromImage(image=img,
                                    scalefactor=1/255, # 1/sigma
                                    size=(self.img_width, self.img_height), 
                                    mean=[0,0,0], #openCV assumes images in BGR, but
                                    # mean value assumes RGB, hence swap RB
                                    swapRB=1,
                                    crop=False)
        self.net.setInput(blob)
        
        # Runs the forward pass to get output of the output layers
        # outs: vectors of length 85 (4 for bounding box, 1 for conf, 80 for class confidence)
        outs = self.net.forward(self.get_layer_names())

        # Remove the bounding boxes with low confidence
        # returns classIds, confidences, bbox_coords
        predictions = self.postprocess(img, outs)
        
        # Put efficiency information. The function getPerfProfile returns the overall time for inference(t) 
        # and the timings for each of the layers(in layersTimes)
        t, _ = self.net.getPerfProfile()
        inference_time = t * 1000.0 / cv.getTickFrequency()
        label = 'Inference time: {%.2f} ms' % inference_time

        print(label)
        cv.putText(img, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

        if plot:
            # cv.imshow('frame', img)
            # cv.waitKey(1)
            plt.figure()
            plt.imshow(img)
            plt.show()  
        # else:
        #     cv.imwrite(f'out_{self.num_imgs_processed}.jpg', img)
        print("PREDDD",predictions)

        return img, inference_time, predictions



    # Get the names of the output layers of network
    def get_layer_names(self):
        # Get the names of all the layers in the network
        layersNames = self.net.getLayerNames()
        # Get the names of the output layers, i.e. the layers with unconnected outputs
        return [layersNames[i[0]-1] for i in self.net.getUnconnectedOutLayers()]

    # Draw the predicted bounding box
    def draw_bboxes(self, frame, classId, conf, bbox_coords):

        left, top, right, bottom = bbox_coords

        # Draw a bounding box.
        cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)

        label = '%.2f' % conf

        # Get the label for the class name and its confidence
        if self.classes:
            assert(classId < len(self.classes))
            label = '%s:%s' % (self.classes[classId], label)

        #Display the label at the top of the bounding box
        labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, labelSize[1])
        cv.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine), (255, 255, 255), cv.FILLED)
        cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1)


    # Remove the bounding boxes with low confidence using non-maxima suppression
    def postprocess(self, frame, outs):
        predictions = [] 

        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]

        classIds = []
        confidences = []
        boxes = []
        # Scan through all the bounding boxes output from the network and keep only the
        # ones with high confidence scores. Assign the box's class label as the class with the highest score.
        classIds = []
        confidences = []
        boxes = []
        for out in outs: # for each elem in tuple
            for detection in out: # for each vector of length 85
                
                scores = detection[5:] # scores for each class


                classId = np.argmax(scores)
                confidence = scores[classId]
                if sum(scores) != 0:
                    print(f'{self.classes[classId]}: {confidence}')
                    

                if confidence > self.conf_threshold:
                    center_x = int(detection[0] * frameWidth)
                    center_y = int(detection[1] * frameHeight)
                    width = int(detection[2] * frameWidth)
                    height = int(detection[3] * frameHeight)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    classIds.append(classId)
                    confidences.append(float(confidence))
                    # only add bounding box if confidence > threshold
                    boxes.append([left, top, width, height])

        # Perform non maximum suppression to eliminate redundant overlapping boxes with
        # lower confidences. 
        indices = cv.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.nms_threshold)
        for i in indices:
            i = i[0]
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            
            bbox_coords = left, top, left+width, top+height
            self.draw_bboxes(frame, classIds[i], confidences[i], bbox_coords)

            predictions.append((self.classes[classIds[i]], confidences[i], bbox_coords))

        return predictions

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Real-time object detection with Yolo-v3')
    # parser.add_argument('--image', help='Path to image file.')
    # parser.add_argument('--video', help='Path to video file.')
    parser.add_argument('--conf_threshold', type=float, default=0.25, help='Confidence threshold for detection')
    parser.add_argument('--nms_threshold', type=float, default=0.4, help='Non-maximal suppression threshold')
    parser.add_argument('--img_width', type=int, default=416, help='Input image width')
    parser.add_argument('--img_height', type=int, default=416, help='Input image height')
    parser.add_argument('--cfg_path', default='./darknet/cfg/', help='Model config directory')
    parser.add_argument('--data_path', default='./darknet/data/', help='Model weights directory')
    parser.add_argument('--model', default='yolov3-tiny', help='Model name')
    parser.add_argument('--classes_file', default='coco.names', help='label')
    parser.add_argument('--webserver_url', default="http://192.168.12.233:81/stream", help='stream url')

    args = parser.parse_args()

    model = Yolov3Tiny(args)
    model.begin_stream()
