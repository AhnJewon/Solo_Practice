from pickletools import optimize
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
from keras import models
from keras import layers
now = datetime.now()
wdf = pd.DataFrame()

for i in range(1967,now.year):
    	
    globals()[f'wd_{i}'] = "./WeatherData_CJ/SURFACE_ASOS_131_DAY_{0}.csv".format(i)
    globals()[f'wdf_{i}'] = pd.read_csv(globals()[f'wd_{i}'], encoding = 'CP949')    
    globals()[f'wdf_{i}'] = globals()[f'wdf_{i}'].rename(columns = {'일시':'date', '평균기온(°C)':'temperature'})
    globals()[f'wdf_{i}']['date'] = pd.to_datetime(globals()[f'wdf_{i}']['date'])
    globals()[f'wdf_{i}']['year'] = globals()[f'wdf_{i}']['date'].dt.year
    globals()[f'wdf_{i}']['month'] = globals()[f'wdf_{i}']['date'].dt.month
    globals()[f'wdf_{i}']['day'] = globals()[f'wdf_{i}']['date'].dt.day
    globals()[f'wdf_{i}'] = globals()[f'wdf_{i}'].loc[:,'temperature':]
    globals()[f'wdf_{i}'] = globals()[f'wdf_{i}'].fillna(0)

for j in range(1967,(now.year-1)):

    wdf = pd.concat([wdf, globals()[f'wdf_{j}']], axis=0)
wdf = wdf.fillna(0)

lastest_year = globals()['wdf_{0}'.format(now.year - 1)]
lastest_year = lastest_year.fillna(0)

wdf.to_csv('wdf.csv')
print(wdf)

feature = wdf.loc[:,'temperature':]

target = wdf[['temperature']]
print(feature.shape)
print(target.shape)
feature_val = lastest_year.loc[:,'temperature':]

target_val = lastest_year[['temperature']]

'''X = tf.keras.layers.Input(shape=[62]) #피쳐의 개수 넣어줌
H = tf.keras.layers.Dense(20, activation='swish')(X)
H1 = tf.keras.layers.Dense(20, activation='swish')(H)
H2 = tf.keras.layers.Dense(20, activation='swish')(H1)
H3 = tf.keras.layers.Dense(20, activation='swish')(H2)
y = tf.keras.layers.Dense(1)(H3)
model = tf.keras.models.Model(X,y)'''

model = models.Sequential()
model.add(layers.Dense(64, activation='swish', input_shape=(feature.shape[1],)))
model.add(layers.Dense(64, activation='swish'))
model.add(layers.Dense(1))


model.compile(optimizer='Adam', loss='mse', metrics=['mae'], run_eagerly=True)

history = model.fit(feature, target, epochs=4, verbose='auto', validation_data=(feature_val, target_val), batch_size=32) #epochs: 학습 횟수, verbose:학습과정 출력 여부

print(model.predict(feature))
print(target_val)

history_dict = history.history

loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1,len(loss) +1)
plt.plot(epochs, loss, 'bo', label = 'Training loss')
plt.plot(epochs, val_loss, 'b', label = 'Validation loss')
plt.title('Teaining and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

plt.clf()
acc = history_dict['acc']
val_acc = history_dict['val_acc']

plt.plot(epochs, acc, 'bo', label = 'Training acc')
plt.plot(epochs, val_acc, 'b', label = 'Validation acc')
plt.title('Teaining and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()

