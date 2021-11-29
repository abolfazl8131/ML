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
#load data
df = pd.read_csv('MSFT.csv', index_col='Date', na_values=['null'], parse_dates=True, infer_datetime_format=True)
print()
print(df.head())
print('contains null:',df.isnull().values.any())
#set a target var

target_var = df['Adj Close']
features = ['Open', 'High', 'Low','Volume']

#scaling data

scaler = MinMaxScaler()
features_transform = scaler.fit_transform(df[features])
features_transform = pd.DataFrame(columns=features, data=features_transform, index = df.index)
print(features_transform.head())

#training test

t_split = TimeSeriesSplit(n_splits=10)
for train_index , test_index in t_split.split(features_transform):
    x_train , x_test = features_transform[:len(train_index)] , features_transform[len(train_index):(len(train_index)+len(test_index))]
    y_train , y_test = target_var[:len(train_index)].values.ravel() , target_var[len(train_index):(len(train_index)+len(test_index))].values.ravel()


#process data using LSTM
trainX = np.array(x_train)
x_train =np.array(x_train)
x_test = np.array(x_test)
x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1])
x_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[1])

#building LSTM Model

model = Sequential()
model.add(LSTM(32, input_shape=(1, trainX.shape[1]), activation='relu', return_sequences=False))

model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
history = model.fit(x_train, y_train, epochs=3000, batch_size=8, verbose=1, shuffle=False)

y_prediction = model.predict(x_test)

"""plt.plot(y_test, label='True value')
plt.plot(y_prediction, label='model value')
plt.xlabel('time')
plt.ylabel('scaled USD')
plt.legend()
plt.show()"""