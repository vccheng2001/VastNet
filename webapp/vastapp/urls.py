from django.urls import path
from vastapp import views
# from django.conf import settings
# from django.conf.urls.static import static

urlpatterns =[ 
    # home

    path('home', views.home, name="home"),
    path('process_image', views.process_image, name = "process_image")  

]


# if settings.DEBUG:
#      urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
