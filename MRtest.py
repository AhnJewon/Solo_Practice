import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split

weather = '/home/u1016/Solo_Practice/OBS_ASOS_TIM_20230324165804.csv'
chungju = pd.read_csv(weather, encoding='cp949')
print(chungju.head)
chungju.head()

chungju = chungju[['기온(°C)']]
print(chungju.head)

forecast_out = 30
chungju[['다음기온']] = chungju[['기온(°C)']].shift(-forecast_out)
print(chungju.tail())

###Create the independent data set (x) #####
#Convert the dataframe to a numpy array

X = np.array(chungju.drop(['다음기온'],1))
X = X[:-forecast_out]
print(X)

Y = np.array(chungju['다음기온'])
Y = Y[:-forecast_out]
print(Y)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
svr_rbf.fit(x_train, y_train)

lr = LinearRegression()
lr.fit(x_train, y_train)

svm_confidence = svr_rbf.score(x_test, y_test)
print("svm confidence: ", svm_confidence)

lr_confidence = lr.score(x_test, y_test)
print("lr confidence: ", lr_confidence)

x_forecast = np.array(chungju.drop(['다음기온'],1))[-forecast_out:]
print(x_forecast)

lr_prediction = lr.predict(x_forecast)
print(lr_prediction)

svm_prediction = svr_rbf.predict(x_forecast)
print(svm_prediction)

chungju =chungju[-forecast_out:]
cj_forecast = chungju.drop(['다음기온','기온(°C)'],1)
cj_forecast['SVM Prediction'] = svm_prediction
cj_forecast['LR Preiction'] = lr_prediction
cj_forecast

import pandas_datareader.data as web
cj_actual = web.DataReader('WDB' ,data_source='yahoo', start='2022-03-25', enf='2022-04-23')
actual = np.array(ch_actual['기온(°C)'])
actual

cj_forecast['actual'] = actual
cj_forecast

import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

plt.figure(figsize=(16, 8))
plt.title('Weather Temperture Comparison of Chungju')
plt.plot(cj_forecast['LR Preiction'])
plt.plot(cj_forecast['SVM Preiction'])
plt.plot(cj_forecast['actual'])
plt.ylabel('Close Temperture(C)', fontsize=18)
plt.legend(['LR Preiction', 'SVM Preiction', 'actual'], loc='lower right')
plt.show()
