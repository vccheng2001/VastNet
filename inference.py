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

parser = argparse.ArgumentParser(description='Real-time object detection with Yolo-v3')
# parser.add_argument('--image', help='Path to image file.')
# parser.add_argument('--video', help='Path to video file.')
parser.add_argument('--conf_threshold', type=float, default=0.25, help='Confidence threshold for detection')
parser.add_argument('--nms_threshold', type=float, default=0.4, help='Non-maximal suppression threshold')
parser.add_argument('--img_width', type=int, default=320, help='Input image width')
parser.add_argument('--img_height', type=int, default=320, help='Input image height')
parser.add_argument('--cfg_path', default='./darknet/cfg/', help='Model config directory')
parser.add_argument('--data_path', default='./darknet/data/', help='Model weights directory')
parser.add_argument('--model', default='yolov3', help='Model name')

args = parser.parse_args()

print('Args: ', args)
# Parse arguments
conf_threshold = args.conf_threshold
nms_threshold = args.nms_threshold
img_width = args.img_width
img_height = args.img_height
cfg_path = args.cfg_path
data_path = args.data_path
model = args.model

# Load names of classes
classesFile = f"{data_path}coco.names"
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# Give the configuration and weight files for the model and load the network using them.
modelConfiguration = f"{cfg_path}{model}.cfg";
modelWeights = f"{cfg_path}{model}.weights"

net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU) 

# Get the names of the output layers
def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i-1] for i in net.getUnconnectedOutLayers()]

# Draw the predicted bounding box
def drawPred(classId, conf, left, top, right, bottom):
    # Draw a bounding box.
    cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)

    label = '%.2f' % conf

    # Get the label for the class name and its confidence
    if classes:
        assert(classId < len(classes))
        label = '%s:%s' % (classes[classId], label)

    #Display the label at the top of the bounding box
    labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine), (255, 255, 255), cv.FILLED)
    cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1)

# Remove the bounding boxes with low confidence using non-maxima suppression
def postprocess(frame, outs):
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
                print(f'{classes[classId]}: {confidence}')
                

            if confidence > conf_threshold:
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
    indices = cv.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    for i in indices:
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        drawPred(classIds[i], confidences[i], left, top, left + width, top + height)

# Process inputs
winName = 'ESP32-CAM Object Detection with Yolo'
cv.namedWindow(winName, cv.WINDOW_NORMAL)

outputFile = f"{model}_out.avi"
# Webcam input
webserver_url="http://192.168.12.233:81/stream"
CAMERA_BUFFER_SIZE=4096
stream=urlopen(webserver_url)
bts=b''


while cv.waitKey(100) < 0:
    chunk = stream.read(CAMERA_BUFFER_SIZE) 
    bts+=chunk
    jpghead=bts.find(b'\xff\xd8')
    jpgend=bts.find(b'\xff\xd9')
    if jpghead>-1 and jpgend>-1:

        jpg=bts[jpghead:jpgend+2]
        bts=bts[jpgend+2:]

        
        
        
        try:
            img=cv.imdecode(np.frombuffer(jpg,dtype=np.uint8),cv.IMREAD_UNCHANGED)
        except:
            continue
        
        v=cv.flip(img,0)
        h=cv.flip(img,1)
        
        frame = img
        h,w=frame.shape[:2]
        frame=cv.resize(frame,(1024,768))
        # blobFromImage performs mean subtraction, scaling, channel swapping
        # creates 4D blob from image
        # resizes, crops image from center
        # subtracts mean values, scales by scalefactor
        # swaps blue, red channels
        blob = cv.dnn.blobFromImage(image=frame,
                                    scalefactor=1/255, # 1/sigma
                                    size=(img_width, img_height), 
                                    mean=[0,0,0], #openCV assumes images in BGR, but
                                    # mean value assumes RGB, hence swap RB
                                    swapRB=1,
                                    crop=False)
        r = blob[0, 0, :, :]

        net.setInput(blob)
        
        # Runs the forward pass to get output of the output layers
        # outs: vectors of length 85 (4 for bounding box, 1 for conf, 80 for class confidence)
        outs = net.forward(getOutputsNames(net))

        # Remove the bounding boxes with low confidence
        postprocess(frame, outs)
        # Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
        t, _ = net.getPerfProfile()
        label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
        print(label)
        cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

        cv.imshow('frame', frame)
        cv.waitKey(1)
