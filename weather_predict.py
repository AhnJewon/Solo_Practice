import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

wd_2021 = '/home/u1016/Solo_Practice/OBS_ASOS_TIM_2021.csv'
wd_2022 = '/home/u1016/Solo_Practice/OBS_ASOS_TIM_2022.csv'

wdf_2021 = pd.read_csv(wd_2021, encoding='cp949')
wdf_2021 = wdf_2021[['일시', '기온(°C)']]
wdf_2021 = wdf_2021.rename(columns = {'일시':'date', '기온(°C)':'temperature'})
wdf_2021['date'] = pd.to_datetime(wdf_2021['date'])
wdf_2021['year'] = wdf_2021['date'].dt.year
wdf_2021['month'] = wdf_2021['date'].dt.month
wdf_2021['day'] = wdf_2021['date'].dt.day
wdf_2021 = wdf_2021[['year', 'month', 'day', 'temperature']]
wdf = pd.concat([wdf_2021])

wdf_2022 = pd.read_csv(wd_2022, encoding='cp949')

md = {}
for i, row in wdf.iterrows():
    m, d, v = (int(row['month']), int(row['day']), float(row['temperature']))
    key = str(m) + '/' + str(d)
    if not(key in md): md[key] = []
    md[key] += [v]

avs = {}
for key in md:
    v = avs[key] = sum(md[key]) / len(md[key])
    print("{0} : {1}".format(key, v))

avs['9/11']

g = wdf.groupby(['month'])["temperature"]

g_avg = g.sum() / g.count()
print(g_avg)

g_avg.plot()
plt.savefig("1y_month-avg.png")
plt.show()

from sklearn.linear_model import LinearRegression

train_month = (wdf['month'] <= 11)
test_month = (wdf['month'] >= 12)
interval = 3

def make_data (data):
    x = []
    y = []
    temps = list(data['temperature'])
    for i in range(len(temps)):
        if i <= interval: continue
        y.append(temps[i])
        xa = []
        for p in range(interval):
            d = i + p - interval
            xa.append(temps[d])
    return (x, y)

train_x, train_y = make_data(wdf[train_month])
test_x, test_y = make_data(wdf[test_month])

lr = LinearRegression(normalize = True)
lr.fit(train_x, train_y)
pre_y = lr.predict(test_x)

plt.figure(figsize=(10,6), dpi = 100)
plt.plot(test_y, c='r')
plt.plot(pre_y, c='b')
plt.savefig('1y_lr.png')
plt.show()
