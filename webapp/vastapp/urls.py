from django.urls import path
from vastapp import views
# from django.conf import settings
# from django.conf.urls.static import static

urlpatterns =[ 
    # login/logout 
    path('login/', views.login_action, name="login"),
    path('register/', views.register_action, name="register"),
    path('logout/', views.logout_action, name="logout"),


    # home
    
    path('home', views.home, name="home"),

    path('profile', views.profile, name="profile"),
    path('update_profile', views.update_profile,name='update_profile'),
    path('get_picture/<int:id>', views.get_picture, name='get_picture'),

    path('process_image', views.process_image, name = "process_image")  

]


# if settings.DEBUG:
#      urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
