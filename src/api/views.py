from django.shortcuts import render, redirect, reverse
from rest_framework import generics
from .serializers import RoomSerializer, SentimentSerializer
from .models import Room, Sentiment
import pandas as pd
import re, sys, json
import numpy as np
import base64
from io import BytesIO
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
from pylab import figure
from .models import Sentiment


from googleapiclient.discovery import build



def Preprocessing(comments):
    preprocessing_result = []
    regex_result = []
    lemmatizing_result = []
    stopword = stopwords.words('english')
    tokenizer = RegexpTokenizer(r'\w+')
    lemmatizer = WordNetLemmatizer()
    
    comments = comments.drop(comments[comments.label == 'Irrelevant'].index)
    comments = comments[comments['label'].notna()]
    comments = comments[comments['text'].notna()]
    
    for line in comments['text']:
        sentence = []
        
        lowercase = line.lower()
        regex = re.sub("[^a-zA-Z0-9 ]+", " ", lowercase)
        regex_result.append(regex)
        tokenize = tokenizer.tokenize(regex)
        
        for word in tokenize:    
            lemma = lemmatizer.lemmatize(word)
            sentence.append(lemma)
    
        lemmatizing_result.append(sentence)
        
        removestopword = (word for word in sentence if word not in stopword)
        join = ' '.join(removestopword)
        preprocessing_result.append(join)
    
    comments['regex'] = regex_result
    comments['lemmatized'] = lemmatizing_result
    comments['preprocessed'] = preprocessing_result
    return comments

def Process(dataFrame, fold):
    accuracies = []
    confusions = []
    tfidfv = TfidfVectorizer(min_df=1,stop_words='english')
    nb = MultinomialNB()
    label_types = dataFrame.label.value_counts(sort=True)
    dataFrame['labelNumber'] = dataFrame['label']
    for i in range(0,len(label_types)):
        dataFrame['labelNumber'] = dataFrame['labelNumber'].replace('{}'.format(label_types.index[i]),'{}'.format(i))
    
    comment = dataFrame['preprocessed']
    label = dataFrame['labelNumber']
    kf = KFold(n_splits=fold)
    for train_index, test_index in kf.split(comment) :
        X_train, X_test = comment[train_index], comment[test_index]
        y_train, y_test = label[train_index], label[test_index]
        comment_train_vector = tfidfv.fit_transform(X_train)
        comment_test_vector = tfidfv.transform(X_test)
        multiNB = nb.fit(comment_train_vector, y_train)
        prediction = nb.predict(comment_test_vector)
        expect = y_test
        accuracy = accuracy_score(expect, prediction)
        confusion = confusion_matrix(expect,prediction, labels=nb.classes_)
        accuracies.append(accuracy)
        confusions.append(confusion)
    result =  {
        'confusions' : confusions,
        'accuracies' : accuracies,
    }

    return result

def get_graph():
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()

    graphic = base64.b64encode(image_png)
    graphic = graphic.decode('utf-8')

    return graphic

# Create your views here.
class RoomView(generics.CreateAPIView):
    queryset = Room.objects.all()
    serializer_class = RoomSerializer

class SentimentView(generics.ListAPIView):
    queryset = Sentiment.objects.all()
    serializer_class = SentimentSerializer

def sentiment_data_upload_view(request):
    return render(request, "example.html",{})

def sentiment_data_view(request):
    if request.method =="POST":
        file_excel = request.FILES['name_file']

        df = pd.read_excel(file_excel , engine='openpyxl')

        for index, row in df.iterrows():
            text = row['comment']
            label = row['Label']
            user = row['author']
            Sentiment(
                text=text,
                label=label,
                user=user
            ).save()

    data = pd.DataFrame(list(Sentiment.objects.all().values()))
    data_column = data.columns.values
    data_numpy = data.to_numpy()


    context =  {
        'data' : data_numpy,
        'data_column' : data_column,
        'next_path': 'data-preprocessing',
    }
    request.session['for_processing'] = data.to_json()
    return render(request, "data_view.html", context)

def sentiment_data_preprocessing(request):
    dictionary = request.session.get('for_processing')
    data_json = json.loads(dictionary)
    data =  pd.DataFrame.from_dict(data_json, orient='columns')    
    data = Preprocessing(data)

    data_column = ['Before Preprocessing','After Preprocessing','label']
    data_numpy = data[['text','preprocessed','label']].head().to_numpy()
    context =  {
        'data' : data_numpy,
        'data_column' : data_column,
        'next_path': 'evaluate',
    }
    request.session['for_evaluation'] = data.to_json()
    return render(request, "data_view.html", context)

def sentiment_evaluate(request, surrogate=0):
    if request.method =="POST":
        fold = request.POST.get('k_amount')
        request.session['fold'] = fold

    dictionary = request.session.get('for_evaluation')
    data_json = json.loads(dictionary)
    data =  pd.DataFrame.from_dict(data_json, orient='columns')

    fold_session = request.session.get('fold')
    process_result = Process(data, int(fold_session))
    #menghitung confusion matrix
    confusions = process_result['confusions']
    accuracyScore = process_result['accuracies']
    
    data_confusion = confusions[surrogate]
    disp = ConfusionMatrixDisplay(confusion_matrix=data_confusion)
    disp = disp.plot(cmap="Blues")

    
    context =  {
        'data' : get_graph(),
        'fold' : range(int(fold_session)),
        'next_path': 'youtube',
    }
    return render(request, "evaluate.html", context)

def sentiment_data_testing(request):
    dictionary = request.session.get('for_testing')
    data_json = json.loads(dictionary)
    data =  pd.DataFrame.from_dict(data_json, orient='columns')  

    comparison = np.where(data['predicted']==data['expected'], "", True)
    data['different'] = comparison

    data_numpy = data.to_numpy()
    data_column = data.columns.values

    context =  {
        'data' : data_numpy,
        'data_column' : data_column,
        'next_path': 'evaluate',
    }
    return render(request, "data_view.html", context)

def sentiment_data_training(request):
    dictionary = request.session.get('for_evaluate')
    data_json = json.loads(dictionary)
    data =  pd.DataFrame.from_dict(data_json, orient='columns')
    data.columns = [''] * len(data.columns)
    data_transpose = data.T
    data_transpose.columns = ['True Neg','False Pos','False Neg','True Pos']
    data_numpy = data_transpose.to_numpy
    data_column = data_transpose.columns.values
    print(data)
    context =  {
        'data' : data_numpy,
        'data_column' : data_column,
        'next_path': 'evaluate',
    }
    return render(request, "evaluate.html", context)

def youtube_api(request, **kwargs):
    comments = []
    api_key = 'AIzaSyCJPLzGVlcC8kLE6b7hDgdSQMHz-rn-hus'
    service = build('youtube','v3',developerKey=api_key)
    channelId='UwsrzCVZAb8'
    results = service.commentThreads().list(part='snippet',videoId=channelId, textFormat='plainText', order='time', maxResults=20).execute()

    for item in results['items']:
        comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
        comments.append(comment)
     
    print(comments)
    context =  {
        'data' : comments
    }
    return render(request, "youtube_api.html", context)

def select_kfold(request):
    return render(request, "kfold.html",{})
