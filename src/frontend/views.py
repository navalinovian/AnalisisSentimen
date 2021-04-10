from django.shortcuts import render
import json
import pandas as pd
import re
import numpy as np
import openpyxl
import sys
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from nltk.tokenize.treebank import TreebankWordDetokenizer

# Create your views here.
def index(request, *args, **kwargs):
    dataJson = "";
    if request.method == "POST":
        file_excel = request.FILES['filename']
        data1 = pd.read_excel(file_excel , engine='openpyxl')
        tokenizer = RegexpTokenizer(r'\w+')
        data1['comment'] = data1['comment'].apply(lambda row: row.lower())
        data1['tokenized'] = data1.apply(lambda row: tokenizer.tokenize(row['comment']), axis=1)
        data1['tokenized'] = data1['tokenized'].apply(lambda row: [re.sub("[^a-zA-Z0-9]+", "", word) for word in row])
        lemmatizer = WordNetLemmatizer()
        data1['lemmatized'] = data1['tokenized'].apply(lambda row: [lemmatizer.lemmatize(word) for word in row])
        stopword = stopwords.words('english')
        data1['removeStopword'] = data1['lemmatized'].apply(lambda row: [word for word in row if word not in stopword])
        irrelevant_data = data1[ data1['Label'] == 'Irrelevant' ].index
        data1.drop(irrelevant_data, inplace = True)
        data1.dropna(subset = ['Label'], inplace=True)
        detoken = TreebankWordDetokenizer()
        data1['detokenized'] = data1['removeStopword'].apply(lambda row: detoken.detokenize(row))
        data1.loc[data1['Label']=='Positive','Label']=1
        data1.loc[data1['Label']=='Negative','Label']=0
        x = data1['detokenized']
        y = data1['Label']
        x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.1,random_state=10)
        tfidfv = TfidfVectorizer(min_df=1,stop_words='english')
        x_trainv = tfidfv.fit_transform(x_train)
        x_testv = tfidfv.transform(x_test)
        y_train = y_train.astype('int')
        nb = MultinomialNB()
        multiNB = nb.fit(x_trainv, y_train)
        pred = nb.predict(x_testv)
        expect = np.array(y_test)
        expect = expect.astype('int')
        confusion = confusion_matrix(expect,pred, labels=[0,1]).ravel()
        accuracyScore = accuracy_score(expect,pred)

        data_isempty = data1.empty
        if data_isempty==False:
            print("True")
        # print(data1.head())
            data = {
                "confusion" : json.dumps(confusion.tolist()),
                "accuracyScore" :accuracyScore
            }
            print(confusion)
            print(accuracyScore)
            dataJson = json.dumps(data)
    context = {
        "data":dataJson
    }
    return render(request, 'frontend/index.html', context)


