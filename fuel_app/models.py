from django.db import models

# Create your models here.
class Dataset(models.Model):
    name = models.CharField(max_length=255)
    file = models.FileField(upload_to="datasets/")
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.file.name

""" 
class User(models.Model):
    username = models.CharField(max_length=50, unique=True)
    email = models.EmailField(unique=True)
    password = models.CharField(max_length=100)  # store hashed in real apps

    def __str__(self):
        return self.username
"""