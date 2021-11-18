from django.urls import path
from vastapp import views

urlpatterns =[ 
    # home

    path('home', views.home, name="home"),
    path('process_image', views.process_image, name = "process_image")  

]