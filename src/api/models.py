from django.db import models
from frontend.views import index
class Sentiment(models.Model):
    text        = models.TextField()
    label       = models.CharField(max_length=8, default="", unique=False)