from django.shortcuts import render, redirect, reverse
from rest_framework import generics
from .serializers import RoomSerializer, SentimentSerializer
from .models import Room, Sentiment
import pandas as pd
import numpy as np
import openpyxl
import re
import json
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize.treebank import TreebankWordDetokenizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix


# Create your views here.
class RoomView(generics.CreateAPIView):
    queryset = Room.objects.all()
    serializer_class = RoomSerializer

class SentimentView(generics.CreateAPIView):
    queryset = Sentiment.objects.all()
    serializer_class = SentimentSerializer

def sentiment_data_upload_view(request):
    return render(request, "example.html",{})

def sentiment_data_view(request):
    if request.method =="POST":
        file_excel = request.FILES['name_file']
        df = pd.read_excel(file_excel , engine='openpyxl')
        data = df.drop(df.columns[0],axis=1)
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
    tokenizer = RegexpTokenizer(r'\w+')

    #menghilangkan label kosong
    data.dropna(subset = ['Label'], inplace=True) 

    #menghilangkan label irrelevant
    irrelevant_data = data[ data['Label'] == 'Irrelevant' ].index
    data.drop(irrelevant_data, inplace = True)

    #lowercase data
    data['comment'] = data['comment'].apply(lambda row: row.lower())

    #tokenisasi data
    data['tokenized'] = data.apply(lambda row: tokenizer.tokenize(row['comment']), axis=1)

    #menghilangkan huruf biasa
    data['tokenized'] = data['tokenized'].apply(lambda row: [re.sub("[^a-zA-Z]+", "", word) for word in row])

    #lemmatisasi perkata
    lemmatizer = WordNetLemmatizer()
    data['lemmatized'] = data['tokenized'].apply(lambda row: [lemmatizer.lemmatize(word) for word in row])

    #menghilangkan stopword
    stopword = stopwords.words('english')
    data['removeStopword'] = data['lemmatized'].apply(lambda row: [word for word in row if word not in stopword])

    #detokenisasi
    detoken = TreebankWordDetokenizer()
    data['detokenized'] = data['removeStopword'].apply(lambda row: detoken.detokenize(row))

    data_column = ['Before Preprocessing','After Preprocessing','Label']
    data_numpy = data[['comment','detokenized','Label']].to_numpy()
    context =  {
        'data' : data_numpy,
        'data_column' : data_column,
        'next_path': 'data-training',
    }
    request.session['for_training'] = data.to_json()
    return render(request, "data_view.html", context)

def sentiment_data_training(request):
    dictionary = request.session.get('for_training')
    data_json = json.loads(dictionary)
    data =  pd.DataFrame.from_dict(data_json, orient='columns')  

    #mengubah label menjadi 1 untuk positif dan 0 untuk negatif
    data.loc[data['Label']=='Positive','Label']=1
    data.loc[data['Label']=='Negative','Label']=0
    x = data['detokenized']
    y = data['Label']

    #membagi jumlah data test sebanyak 10%
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.1)
    y_train = y_train.astype('int')
    expect = np.array(y_test)
    expect = expect.astype('int')   

    #melakukan pembobotan tfidf
    tfidfv = TfidfVectorizer(min_df=1,stop_words='english')
    x_trainv = tfidfv.fit_transform(x_train)
    x_testv = tfidfv.transform(x_test)

    #melakukan training menggunakan multinomial naive bayes
    nb = MultinomialNB()
    multiNB = nb.fit(x_trainv, y_train) 

    #melakukan prediksi
    pred = nb.predict(x_testv)

    #menghitung confusion matrix
    confusion = confusion_matrix(expect,pred, labels=[0,1]).ravel()
    accuracyScore = accuracy_score(expect,pred)
    
    tfidf_output = pd.DataFrame(x_trainv.toarray(), columns = tfidfv.get_feature_names())
    data_numpy = tfidf_output.head().to_numpy()
    data_column = tfidf_output.columns.values
    
    context =  {
        'data' : data_numpy,
        'data_column' : data_column,
        'next_path': 'data-testing',
    }
    result = pd.DataFrame({'predicted':pred, 'expected':expect})
    confusion = pd.DataFrame({'confusion':confusion})
    request.session['for_testing'] = result.to_json()
    request.session['for_evaluate'] = confusion.to_json()
    return render(request, "data_view.html", context)

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

def sentiment_evaluate(request):
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

