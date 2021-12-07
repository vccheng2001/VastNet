from django.shortcuts import render, redirect
from django.urls import reverse
from django.http import HttpResponse, Http404
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt
from .forms import UserImageForm  
from .models import UploadImage, Capture

# Download the helper library from https://www.twilio.com/docs/python/install
import os
from twilio.rest import Client
from django.core.files.base import ContentFile

from PIL import Image
import io
import sys
import cv2 as cv
import base64

import io
sys.path.insert(0, '../') # add 
from inference import Yolov3Tiny, Args
import numpy as np
import datetime 
# Create your views here.
def home(request):
    context = {}
    print('in home view')

	
    return render(request, "vastapp/home.html", context=context)


#####################################################################

@csrf_exempt
def process_image(request):  
    context = {}
    
    if request.method == 'POST':  

        if request.FILES.get("image", None) is not None:

            # # Find your Account SID and Auth Token at twilio.com/console
            # # and set the environment variables. See http://twil.io/secure
            # account_sid = 'AC147174888cada5d58041821c8e477d2c'
            # auth_token = '353ed3df37e886ef162fd4376040ba6c'
            # client = Client(account_sid, auth_token)

            # message = client.messages \
            #                 .create(
            #                     body="VAST-Net noticed something unusual, deploy the drone!",
            #                     from_='+18285547879',
            #                     to='+16262176595'
            #                 )

            # print(message.sid)

                
            
            img_file = request.FILES["image"].read()
            # decode bytes into img 
            img = cv.imdecode(np.fromstring(img_file, np.uint8), cv.IMREAD_COLOR)
            # yolo detect 
            img_with_annot, inference_time, predictions = yolo_detect(img)

            if len(predictions) > 0:


                # save as jpg in captures/ folder 
                ret, buf = cv.imencode('.jpg', img_with_annot)
                content = ContentFile(buf.tobytes())

                # create new Capture
                cap = Capture()
                cap.inference_time = inference_time
                now =  datetime.datetime.now().isoformat()
                cap.image.save(f'{now}.jpg', content) # save 

                # save highest confidence class
                cap.pred_1 = predictions[0][0]
                cap.conf_1 = round(predictions[0][1] * 100, 2)

                try:
                    cap.pred_2 = predictions[1][0]
                    cap.conf_2 = round(predictions[1][1] * 100, 2)
                except: pass 

                try:
                    cap.pred_3 = predictions[2][0]
                    cap.conf_3 = round(predictions[2][1] * 100, 2)
                except: pass


                context["captures"] = Capture.objects.all()
                cap.save()
            
                return render(request, 'vastapp/process_image.html', context=context)

            else:
                context["captures"] = Capture.objects.all()
                return render(request, 'vastapp/process_image.html', context=context)  

    context["captures"] = Capture.objects.all()
    return render(request, 'vastapp/process_image.html', context=context)  



def yolo_detect(img):
    args = Args()
    args.data_path = "../darknet/data/"
    args.cfg_path = "../darknet/cfg/"
    model = Yolov3Tiny(args)
    args.custom_weights = './darknet/small.weights'

    
    img_with_annot, inference_time, predictions = model.predict_img(img, plot=False)

    
    # print(inference_time, predictions)
    return img_with_annot, inference_time, predictions