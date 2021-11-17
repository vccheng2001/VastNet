from django.urls import path
from vastapp import views

urlpatterns =[ 
    # home

    path('home', views.home, name="home"),
    path('', views.image_request, name = "image-request")  

]