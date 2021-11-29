# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 10:36:04 2021

@author: abolfazl81
"""
# crypto prediction

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sc
from sklearn.preprocessing import MinMaxScaler
import pandas_datareader as web
import datetime as dt
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential
print(np.__version__)
c_c = 'BTC'
against_currency = 'USD'
start = dt.datetime(2016, 1, 1)
end = dt.datetime.now()

data = web.get_data_yahoo(f'{c_c}-{against_currency}', start, end)
print(data)
scaler = MinMaxScaler(feature_range=(0,1))

scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1,1))
print(scaled_data)
prediction_days = 60
future_days = 30
x_train, y_train = [], []
for x in range(prediction_days, len(scaled_data)):
    x_train.append(scaled_data[x-prediction_days: x, 0])
    y_train.append(scaled_data[x,0])
    
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    

# NN

model = Sequential()

model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=35, batch_size=32)


#testing the model

test_start = dt.datetime(2021, 1, 1)
test_end = dt.datetime.now()

test_data = web.get_data_yahoo(f'{c_c}-{against_currency}', test_start, test_end)


actual_prices = test_data['Close'].values

total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)

inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
inputs = inputs.reshape(-1,1)
inputs = scaler.fit_transform(inputs)

x_test = []

for x in range(prediction_days, len(inputs)):
    x_test.append(inputs[x-prediction_days:x, 0 ])
    
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

prediction_prices = model.predict(x_test)
prediction_prices = scaler.inverse_transform(prediction_prices)

plt.plot(actual_prices, label='actual', color='black')
plt.plot(prediction_prices, label='prediction', color='red')

plt.xlabel('time')
plt.ylabel('price')

plt.show()

real_data = [inputs[len(inputs) + 1 - prediction_days : len(inputs)+2 ,0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))
prediction = model.predict(real_data)
prediction = scaler.inverse_transform(prediction)
print(prediction)

























