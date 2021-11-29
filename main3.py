import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sc
from sklearn.preprocessing import MinMaxScaler
import pandas_datareader as web
import datetime as dt
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential
import keras.backend as k
from keras.callbacks import EarlyStopping
from keras.utils.vis_utils import plot_model
from sklearn.model_selection import TimeSeriesSplit
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = load_breast_cancer(as_frame=True, return_X_y=True)

df = pd.read_csv('breast-cancer_csv.csv')

X = df.iloc[:,:10]
y = df.iloc[:,9]


LabelEncoder_y = LabelEncoder()
y = LabelEncoder_y.fit_transform(y)

X = OneHotEncoder().fit_transform(X).toarray()

X_train, X_test, y_train , y_test = train_test_split(X,y, random_state=None)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

############ machine learning

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)






prediction = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm2 = confusion_matrix(y_test, prediction)




















