from django.db import models  
from django_base64field.fields import Base64Field
from datetime import datetime 
from django.contrib.auth.models import User


class Profile(models.Model):  
    user = models.OneToOneField(User, on_delete = models.CASCADE)  
    picture = models.FileField(blank=True)
    bio = models.TextField(null=True)
    # pic content type
    content_type = models.CharField(max_length=50,null=True)

    
    def __str__(self):  
        return f"{self.user}'s' profile" 
# image capture 
class Capture(models.Model):  
    inference_time = models.FloatField(null=True, blank=True)
    image = models.ImageField(upload_to='captures/',null=True, blank=True)
    
    pred_1 = models.CharField(null=True, blank=True, max_length=50)
    conf_1 = models.FloatField(null=True, blank=True)

    pred_2 = models.CharField(null=True, blank=True, max_length=50)
    conf_2 = models.FloatField(null=True, blank=True)

    pred_3 = models.CharField(null=True, blank=True, max_length=50)
    conf_3 = models.FloatField(null=True, blank=True)

    timestamp = models.DateTimeField(default=datetime.now, blank=True)


