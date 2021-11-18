from django.shortcuts import render, redirect
from django.urls import reverse
from django.http import HttpResponse, Http404
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt
from .forms import UserImageForm  
from .models import UploadImage, Capture
from PIL import Image
import io
import sys
import cv2 as cv
import base64


sys.path.insert(0, '../') # add 
from inference import Yolov3Tiny, Args
import numpy as np

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
            img_file = request.FILES["image"].read()

            #read image file string data
            #convert string data to numpy array
            # convert numpy array to image
            img= cv.imdecode(np.fromstring(img_file, np.uint8), cv.IMREAD_COLOR)
            
            img_with_annot, inference_time, predictions = yolo_detect(img)


            img_b64 = str(base64.b64encode(img_with_annot), 'utf8')
            dec = base64.b64decode(img_b64)
            print(dec)

            cap = Capture.objects.create(image=img_b64, inference_time=inference_time)
            cap.save()


            context["captures"] = Capture.objects.all()
            return render(request, 'vastapp/process_image.html', context=context)
    context["captures"] = Capture.objects.all()
    return render(request, 'vastapp/process_image.html', context=context)  



def yolo_detect(img):
    args = Args()
    args.data_path = "../darknet/data/"
    args.cfg_path = "../darknet/cfg/"
    model = Yolov3Tiny(args)
    
    img_with_annot, inference_time, predictions = model.predict_img(img, plot=False)


    print(inference_time, predictions)
    return img_with_annot, inference_time, predictions