from django.shortcuts import render, redirect
from django.urls import reverse
from django.http import HttpResponse, Http404
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt
from .forms import ProfileForm
from .models import Capture, Profile
import random
import plotly 
from plotly.offline import download_plotlyjs, plot
import plotly.graph_objs as go

import plotly.express as px
import numpy
from django.utils import timezone
import pytz

# user login/logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login, logout

# forms 
from vastapp.forms import LoginForm, RegisterForm



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


def filter_predictions(predictions):
    seen = set()      #Check Flag
    filtered = []
    for p in predictions:
        if p[0] not in seen:
            filtered.append(p)
            seen.add(p[0])
    return filtered

def to_percent(x):
    return round(x*100, 2)

# Create your views here.
@login_required
def home(request):
    context = {}

    all_captures = list(Capture.objects.all())
    if len(all_captures) > 9:
        rand_captures = random.sample(all_captures, 9)
    else:
        rand_captures = all_captures 

    # top predictions
    all_preds = Capture.objects.values_list('pred_1', flat = True)
    # get frequency of classes_predicted, store in pie chart 
    counts_dict = get_counts(all_preds)
    freqs_dict = {k: v / total for total in (sum(counts_dict.values()),) for k, v in counts_dict.items()}


    if all_preds != []: 
        # display pie chart 
        fig = px.pie(values=counts_dict.values(), names=counts_dict.keys())
        background_color = 'black'
        fig.update_layout(
            hovermode="closest",
            plot_bgcolor=background_color,
            paper_bgcolor=background_color,
            font=dict(color= 'white', size=24)
        )
        graph = fig.to_html(full_html=False, default_height=500, default_width=700)

    else:
        graph = None

    try:
        most_freq_animal = max(counts_dict, key=counts_dict.get)
        warning_msg = f'Caution! Detecting a high number of {most_freq_animal}s in the area'
   
   
    
    except:
        print('No captures yet!')
        most_freq_animal = None
        warning_msg = ''
        







    # Using plotly.express
    all_timestamps = Capture.objects.values_list('timestamp', flat = True)

    if all_timestamps != []:
        try: 
            # time_graph = None
            df = px.data.stocks()
            fig = px.line(df, x=all_timestamps, y=[random.randrange(0,7) for _ in range(len(all_timestamps))])
            fig.update_layout(
                hovermode="closest",
                plot_bgcolor=background_color,
                paper_bgcolor=background_color,
                font=dict(color= 'white', size=24)
            )
            time_graph = fig.to_html(full_html=False, default_height=500, default_width=1200)
        except:
            time_graph = None
    else:
        time_graph = None
        graph = None
    # fig.show()


    context = {'graph': graph, 
                'time_graph':time_graph,
              'counts_dict':counts_dict,
              'rand_captures':rand_captures, 
              'most_freq_animal':most_freq_animal,
              'warning_msg':warning_msg}
    return render(request, "vastapp/home.html", context=context)


def plot_with_style(fig):
    plot_div = plotly.offline.plot(fig, output_type='div')
    template = """
    <head>
    <body style="background-color:#111111;">
    </head>
    <body>
    {plot_div:s}
    </body>""".format(plot_div = plot_div)
    return template 




# Gets dictionary of counts of each captured animal
def get_counts(all_preds):
    top_preds_lst = list(all_preds)
    counts_dict = {}
    for i in top_preds_lst:
        counts_dict[i] = counts_dict.get(i, 0) + 1
    return counts_dict




#####################################################################
#                     User Profile
####################################################################
@login_required
def profile(request):
    context={}
    profile = Profile.objects.get(user=request.user)
    # retrieve user's profile 
    context["profile"] = profile
    # retrieve form associated with user's profile 
    context["form"] = ProfileForm(instance=profile)

    return render(request, 'vastapp/profile.html', context=context)

# user can update their own profile page 
@login_required
def update_profile(request):
    context = {}
    profile = Profile.objects.get(user=request.user) # logged in user 
    # POST method 
    form = ProfileForm(request.POST, request.FILES, instance=profile)
    if not form.is_valid():
        context['form'] = form 
    else:
        # if form is valid, save cleaned data         
        picture = form.cleaned_data["picture"]
        profile.content_type = picture.content_type
        profile.bio = form.cleaned_data["bio"]
        profile.save()
        form.save()

    # put into context
    context["form"] = form
    context["profile"] = profile
    return render(request, 'vastapp/profile.html', context)
#####################################################################

@csrf_exempt #login_required
def process_image(request):  
    context = {}
    
    if request.method == 'POST':  

        if request.FILES.get("image", None) is not None:


                
            
            img_file = request.FILES["image"].read()
            # decode bytes into img 
            img = cv.imdecode(np.fromstring(img_file, np.uint8), cv.IMREAD_COLOR)
            # yolo detect 


            args = Args()
            args.conf_threshold = 0.85
            args.weights = "../darknet/small.weights"
            args.classes_file = '../darknet/small_cfg/small.names'
            args.cfg = "../darknet/small_cfg/small.cfg"

            frame, inference_time, predictions = yolo_detect(img, args)


            # cv.imshow("predicted", frame)
            # key = cv.waitKey(250)
            # if key == 27:#if ESC is pressed, exit loop
            #     cv.destroyAllWindows()



            if len(predictions) > 0:
                print('predicted!')

                predictions = filter_predictions(predictions)

                # save as jpg in captures/ folder 
                ret, buf = cv.imencode('.jpg', frame)
                content = ContentFile(buf.tobytes())

                # create new Capture
                cap = Capture()
                cap.inference_time = round(inference_time,3)

                                

                cap.timestamp = datetime.datetime.now()
                cap.image.save(f'{cap.timestamp}.jpg', content) # save 


                # save highest confidence class
                cap.pred_1 = predictions[0][0]
                cap.conf_1 = to_percent(predictions[0][1])

                try:
                    cap.pred_2 = predictions[1][0]
                    cap.conf_2 = to_percent(predictions[1][1])
                except: pass 

                try:
                    cap.pred_3 = predictions[2][0]
                    cap.conf_3 = to_percent(predictions[2][1])
                except: pass



                # Find your Account SID and Auth Token at twilio.com/console
                # and set the environment variables. See http://twil.io/secure

                account_sid = 'AC420eee3ab4a46c49ad21686cf18ad554'
                auth_token = 'bf0ffbf9ce9bacd05d7bbcc41ba0f41e'
                client = Client(account_sid, auth_token)

                message = client.messages \
                                .create(
                                    body=f"[{cap.timestamp}] VAST-Net noticed a {cap.pred_1.lower()}!",
                                    from_='+12077050576',
                                    to='+14088312252'
                                )                

                context["new_cap"] = cap

            
                context["captures"] = Capture.objects.all()

                # warning_msg = None
                # if cap.pred_1 in ["bear", "bobcat"]:
                #     warning_msg = f"Caution! A {cap.pred_1} has been detected."
                # context["warning_msg"] = warning_msg
                cap.save()
            
                return render(request, 'vastapp/process_image.html', context=context)

            else:
                context["captures"] = Capture.objects.all()
                return render(request, 'vastapp/process_image.html', context=context)  

    context["captures"] = Capture.objects.all()
    return render(request, 'vastapp/process_image.html', context=context)  



def yolo_detect(img, args):
    model = Yolov3Tiny(args)

    
    frame, inference_time, predictions = model.predict_img(img, plot=False)

    
    # print(inference_time, predictions)
    return frame, inference_time, predictions



#####################################################################
#                   User Profile Picture 
####################################################################

# retrieves and renders profile picture
@login_required
def get_picture(request, id):
    profile = Profile.objects.get(id=id)
    # raise error if no picture field 
    if not profile.picture:
        raise Http404

    # display picture
    return HttpResponse(profile.picture, content_type=profile.content_type)


#####################################################################
#                     User Registration
####################################################################
def register_action(request):
    context = {}

    # If GET request, display blank registration form
    if request.method == 'GET':
        context['form'] = RegisterForm()
        return render(request, 'vastapp/register.html', context)

    # If POST request, validate the form
    form = RegisterForm(request.POST)
    context['form'] = form

    # Validates the form.
    if not form.is_valid():
        return render(request, 'vastapp/register.html', context)


    # Register and login new user.
    new_user = User.objects.create_user(username=form.cleaned_data['username'], 
                                        password=form.cleaned_data['password'],
                                        email=form.cleaned_data['email'],
                                        first_name=form.cleaned_data['first_name'],
                                        last_name=form.cleaned_data['last_name'])
    # Saves new user profile, authenticate 
    new_user.save()
    new_user = authenticate(username=form.cleaned_data['username'],
                            password=form.cleaned_data['password'])


    # create new profile for user 
    new_profile = Profile.objects.create(user=new_user)
    # new_profile.countries_visited+=[country]
    new_profile.save()

    # Login
    login(request, new_user)
    return redirect(reverse('home'))

#####################################################################
#                     Logout action
####################################################################
def logout_action(request):
    logout(request)
    return redirect(reverse('login'))


#####################################################################
#                           Login
####################################################################
def login_action(request):

    context = {}

    # If GET request
    if request.method == 'GET':
        # If user already logged in, go to global stream page 
        if request.user.is_authenticated:
            return redirect(reverse('home'))  
        else:
            context['form'] = LoginForm()
            context["title"] = "Login"
            return render(request, 'vastapp/login.html', context)
        

    # If POST request, validate login form 
    form = LoginForm(request.POST)
    context['form'] = form

    # If not valid form 
    if not form.is_valid():
        print('Invalid login')
        return render(request, 'vastapp/login.html', context)

    # Authenticate user and log in
    username = form.cleaned_data['username']
    password = form.cleaned_data['password']
    user = authenticate(username=username, password=password)
    login(request, user)
    return redirect('/')
