# -*- coding: utf-8 -*-
"""
Created on Thu May 31 17:38:22 2018

@author: nEW u
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


corpus = []
dataset = pd.read_csv('twitter.csv')
dataset['existence'] = dataset['existence'].replace('Y','Yes')
dataset['existence'] = dataset['existence'].replace('N','No')


for i in range(0,6090):
    if str(dataset['existence'][i]) != "nan":
        corpus.append(dataset['tweet'][i])


import re
import nltk
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


d=[]


#textual filteration
for i in range(0,4225):
    review = re.sub('[^a-zA-Z]',' ',corpus[i])
    review = review.lower()
    review = review.split()
    #we will have to remove articles and prepositions
    from nltk.corpus import stopwords

    review = [ps.stem(word) for word in review if word not in set(stopwords.words('english'))]
    review = ' '.join(review)
    d.append(review)


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 4100)
X = cv.fit_transform(d).toarray()
Y = []


for i in range(0,6090):
    if str(dataset['existence'][i]) != "nan":
        Y.append(dataset['existence'][i])


from sklearn.preprocessing import LabelEncoder
labenc_Y= LabelEncoder()
Y = labenc_Y.fit_transform(Y)


from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,train_size=0.9,random_state=0)#preferably random state value is used as 0


from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)

#applying the Naive Bayes model
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=25,criterion='entropy',random_state=0)
classifier.fit(X_train,Y_train)

#predict result

y_pred = classifier.predict(X_test)


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,y_pred)

#83.215% ACCURACY