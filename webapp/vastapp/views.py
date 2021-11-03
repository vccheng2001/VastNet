from django.shortcuts import render, redirect
from django.urls import reverse
from django.http import HttpResponse, Http404
from django.conf import settings

# Create your views here.
def home(request):
    context = {}
    print('in home view')

	
    return render(request, "vastapp/home.html", context=context)
