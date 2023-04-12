from pickletools import optimize
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime

now = datetime.now()
wdf = pd.DataFrame()

for i in range(1904,now.year):
    	
    globals()[f'wd_{i}'] = "C:/Users/inno5/Source/Repos/AhnJewon/Solo_Practice/weather_data_CB/OBS_ASOS_DD_{0}.csv".format(i)
    globals()[f'wdf_{i}'] = pd.read_csv(globals()[f'wd_{i}'], encoding = 'CP949')    
    globals()[f'wdf_{i}'] = globals()[f'wdf_{i}'].rename(columns = {'일시':'date', '평균기온(°C)':'temperature'})
    globals()[f'wdf_{i}']['date'] = pd.to_datetime(globals()[f'wdf_{i}']['date'])
    globals()[f'wdf_{i}']['year'] = globals()[f'wdf_{i}']['date'].dt.year
    globals()[f'wdf_{i}']['month'] = globals()[f'wdf_{i}']['date'].dt.month
    globals()[f'wdf_{i}']['day'] = globals()[f'wdf_{i}']['date'].dt.day
    globals()[f'wdf_{i}'] = globals()[f'wdf_{i}'][['year','month','day','temperature','평균 풍속(m/s)','최대 순간 풍속(m/s)','최대 풍속(m/s)','평균 이슬점온도(°C)','평균 증기압(hPa)','평균 현지기압(hPa)','평균 해면기압(hPa)','가조시간(hr)','평균 전운량(1/10)','평균 중하층운량(1/10)','평균 지면온도(°C)','최저 초상온도(°C)']]

    wdf = pd.concat([wdf, globals()[f'wdf_{i}']], axis=0)

lastest_year = globals()['wdf_{0}'.format(now.year - 1)]

wdf = wdf.dropna()

print(wdf)

feature = wdf[['year','month','day','평균 풍속(m/s)','최대 순간 풍속(m/s)','최대 풍속(m/s)','평균 이슬점온도(°C)','평균 증기압(hPa)','평균 현지기압(hPa)','평균 해면기압(hPa)','가조시간(hr)','평균 전운량(1/10)','평균 중하층운량(1/10)','평균 지면온도(°C)','최저 초상온도(°C)']]
target = wdf[['temperature']]

feature_val = lastest_year[['year','month','day','평균 풍속(m/s)','최대 순간 풍속(m/s)','최대 풍속(m/s)','평균 이슬점온도(°C)','평균 증기압(hPa)','평균 현지기압(hPa)','평균 해면기압(hPa)','가조시간(hr)','평균 전운량(1/10)','평균 중하층운량(1/10)','평균 지면온도(°C)','최저 초상온도(°C)']]

target_val = lastest_year[['temperature']]



X = tf.keras.layers.Input(shape=[15]) #feature 개수
H = tf.keras.layers.Dense(1024, activation='relu')(X) #hidden layer node 개수:20 layer 개수는 본 코드(H = ...)의 수
H1 = tf.keras.layers.Dense(256, activation='relu')(H)
H2 = tf.keras.layers.Dense(128, activation='relu')(H1)
H3 = tf.keras.layers.Dense(64, activation='relu')(H2)
H4 = tf.keras.layers.Dense(16, activation='relu')(H3)
'''H5 = tf.keras.layers.Dense(16, activation='relu')(H4)
H6 = tf.keras.layers.Dense(16, activation='relu')(H5)'''
Y = tf.keras.layers.Dense(1, activation='softmax')(H2)
model = tf.keras.models.Model(X,Y) 
model.compile(optimizer='nadam', loss='msle', metrics=['acc'], run_eagerly = True)


history = model.fit(feature, target, epochs=100, verbose='auto', validation_data=(feature_val, target_val)) #epochs: 학습 횟수, verbose:학습과정 출력 여부

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

