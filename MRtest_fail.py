import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split

weather = './OBS_ASOS_TIM_2021.csv' 

chungju = pd.read_csv(weather, encoding='cp949')
print(chungju.head)
chungju.head()

chungju = chungju[['기온(°C)']]
print(chungju.head)

forecast_out = 25
chungju[['다음기온']] = chungju[['기온(°C)']].shift(-forecast_out)
print(chungju.tail())

###Create the independent data set (x) #####
#Convert the dataframe to a numpy array

X = np.array(chungju.drop(['다음기온'],axis=1))
X = X[:-forecast_out]
print(X)

Y = np.array(chungju['다음기온'])
Y = Y[:-forecast_out]
print(Y)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

svr_rbf = SVR(kernel='rbf', C=100, gamma=1, epsilon = 0.2)
svr_rbf.fit(x_train, y_train)

lr = LinearRegression()
lr.fit(x_train, y_train)

svm_confidence = svr_rbf.score(x_test, y_test)
print("svm confidence: ", svm_confidence)

lr_confidence = lr.score(x_test, y_test)
print("lr confidence: ", lr_confidence)

x_forecast = np.array(chungju.drop(['다음기온'],axis=1))[-forecast_out:]
print(x_forecast)

lr_prediction = lr.predict(x_forecast)
print(lr_prediction)

svm_prediction = svr_rbf.predict(x_forecast)
print(svm_prediction)

chungju =chungju[-forecast_out:]
cj_forecast = chungju.drop(['다음기온','기온(°C)'],axis=1)
cj_forecast['SVM Prediction'] = svm_prediction
cj_forecast['LR Prediction'] = lr_prediction
cj_forecast

weather2 = './OBS_ASOS_TIM_2021.csv'
ch_actual = pd.read_csv(weather2, encoding='cp949')
ch_actual.head()
ch_actual = ch_actual[['기온(°C)']]
ch_actual = ch_actual[-27:-2]

actual = np.array(ch_actual['기온(°C)'])
actual

cj_forecast['actual'] = actual
cj_forecast
print(cj_forecast)

import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

plt.figure(figsize=(16, 8))
plt.title('Weather Temperture Comparison of Chungju')
plt.plot(cj_forecast['LR Prediction'])
plt.plot(cj_forecast['SVM Prediction'])
plt.plot(cj_forecast['actual'])
plt.ylabel('Close Temperture(C)', fontsize=18)
plt.legend(['LR Preiction', 'SVM Preiction', 'actual'], loc='lower right')
plt.show()
