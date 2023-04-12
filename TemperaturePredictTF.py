from pickletools import optimize
import pandas as pd
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
    globals()[f'wdf_{i}'] = globals()[f'wdf_{i}'][['year','month','day','temperature','최저기온(°C)','최저기온 시각(hhmi)','최고기온(°C)','최고기온 시각(hhmi)','평균 풍속(m/s)','최대 순간 풍속(m/s)','최대 풍속(m/s)','평균 이슬점온도(°C)','평균 증기압(hPa)','평균 현지기압(hPa)','평균 해면기압(hPa)','가조시간(hr)','평균 전운량(1/10)','평균 중하층운량(1/10)','평균 지면온도(°C)','최저 초상온도(°C)']]

    wdf = pd.concat([wdf, globals()[f'wdf_{i}']], axis=0)

lastest_year = globals()['wdf_{0}'.format(now.year - 1)]

wdf = wdf.dropna()
wdf.to_csv('wdf.csv')
print(wdf)

feature = wdf[['year','month','day','temperature','최저기온(°C)','최저기온 시각(hhmi)','최고기온(°C)','최고기온 시각(hhmi)','평균 풍속(m/s)','최대 순간 풍속(m/s)','최대 풍속(m/s)','평균 이슬점온도(°C)','평균 증기압(hPa)','평균 현지기압(hPa)','평균 해면기압(hPa)','가조시간(hr)','평균 전운량(1/10)','평균 중하층운량(1/10)','평균 지면온도(°C)','최저 초상온도(°C)']]
target = wdf[['temperature']]

feature_val = lastest_year[['year','month','day','temperature','최저기온(°C)','최저기온 시각(hhmi)','최고기온(°C)','최고기온 시각(hhmi)','평균 풍속(m/s)','최대 순간 풍속(m/s)','최대 풍속(m/s)','평균 이슬점온도(°C)','평균 증기압(hPa)','평균 현지기압(hPa)','평균 해면기압(hPa)','가조시간(hr)','평균 전운량(1/10)','평균 중하층운량(1/10)','평균 지면온도(°C)','최저 초상온도(°C)']]
target_val = lastest_year[['temperature']]

model = models.Sequential() 
model.add(layers.Dense(1024, activation='swish', input_shape=(20,)))
model.add(layers.Dense(512, activation='swish'))
model.add(layers.Dense(128, activation='swish'))
model.add(layers.Dense(64, activation='swish'))
model.add(layers.Dense(32, activation='swish'))
model.add(layers.Dense(16, activation='swish'))
model.add(layers.Dense(8, activation='swish'))
model.add(layers.Dense(4, activation='swish'))
model.add(layers.Dense(1, activation='softmax'))

tf.keras.optimizers.experimental.Nadam(
    learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07,
    weight_decay=None,
    clipnorm=None,
    clipvalue=None,
    global_clipnorm=None,
    use_ema=False,
    ema_momentum=0.99,
    ema_overwrite_frequency=None,
    jit_compile=True,
    name='Nadam'
    
)

model.compile(optimizer='Nadam', loss='mse', metrics=['acc'])

history = model.fit(feature, target, epochs=4, verbose='auto', validation_data=(feature_val, target_val), batch_size=20) #epochs: 학습 횟수, verbose:학습과정 출력 여부

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

