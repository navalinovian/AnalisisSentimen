from django.urls import path
from .views import RoomView, SentimentView, sentiment_data_upload_view 
from .views import sentiment_data_view, sentiment_data_preprocessing, sentiment_evaluate, youtube_api

urlpatterns = [
    path('home', RoomView.as_view()),
    path('sentiment', SentimentView.as_view()),
    path('upload', sentiment_data_upload_view),
    path('data-view', sentiment_data_view, name='data-view'),
    path('data-preprocessing', sentiment_data_preprocessing, name='data_preprocessing'),
    path('evaluate', sentiment_evaluate),
    path('evaluate/<int:surrogate>/', sentiment_evaluate),
    path('youtube-api', youtube_api),
]