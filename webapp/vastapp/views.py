from django.shortcuts import render, redirect
from django.urls import reverse
from django.http import HttpResponse, Http404
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt
from .forms import UserImageForm  
from .models import UploadImage, Capture

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
                
            
            img_file = request.FILES["image"].read()
            # decode bytes into img 
            img = cv.imdecode(np.fromstring(img_file, np.uint8), cv.IMREAD_COLOR)
            # yolo detect 
            img_with_annot, inference_time, predictions = yolo_detect(img)

            # img_b64 = base64.b64encode(img_with_annot).decode('utf-8')

            ret, buf = cv.imencode('.jpg', img_with_annot) 
            content = ContentFile(buf.tobytes())


            cap = Capture()
            now =  datetime.datetime.now().isoformat()

            cap.image.save(f'{now}.jpg', content)
            

            context["captures"] = Capture.objects.all()
            # context['img_b64'] = img_file
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