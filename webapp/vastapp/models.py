from django.db import models  
from django_base64field.fields import Base64Field

  
class UploadImage(models.Model):  
    label = models.CharField(max_length=200)  
    image = models.ImageField(upload_to='images')  
  
    def __str__(self):  
        return self.label

# image capture 
class Capture(models.Model):  
    # image = models.FileField(blank=True)
    inference_time = models.CharField(max_length=50,null=True)

    image = Base64Field(max_length=900000, blank=True, null=True)
