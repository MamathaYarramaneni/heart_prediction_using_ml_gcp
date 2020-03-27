#!/usr/bin/python

import numpy as np
import pandas as pd

file=open("heart.csv",'r')
dataset=pd.read_csv(file)

X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,-1].values
dataset

from sklearn.model_selection import train_test_split as tts
X_train,X_test,Y_train,Y_test=tts(X,Y,test_size=0.2,random_state=0)
X_test,Y_test

from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.fit_transform(X_test)
X_train

from sklearn.naive_bayes import GaussianNB
nbclassifier=GaussianNB()
nbclassifier.fit(X_train,Y_train)

Y_pred = nbclassifier.predict(X_test)
Y_pred

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Y_test,Y_pred)
print("confusion matrix for Naive bayes prediction: ",cm)

from sklearn import metrics
print("Accuracy for Naive Bayes prediction:",metrics.accuracy_score(Y_test, Y_pred))

print("Precision for Naive Bayes prediction:",metrics.precision_score(Y_test, Y_pred,average=None))
