from django.urls import path
from .views import RoomView, SentimentView, sentiment_data_upload_view, sentiment_data_view, sentiment_data_preprocessing, sentiment_data_training, sentiment_data_testing, sentiment_evaluate

urlpatterns = [
    path('home', RoomView.as_view()),
    path('sentiment', SentimentView.as_view()),
    path('upload', sentiment_data_upload_view),
    path('data-view', sentiment_data_view, name='data-view'),
    path('data-preprocessing', sentiment_data_preprocessing, name='data_preprocessing'),
    path('data-training', sentiment_data_training),
    path('data-testing', sentiment_data_testing),
    path('evaluate', sentiment_evaluate),
]