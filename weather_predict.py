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
wdf = pd.contcat([wdf2021])

wdf_2022 = pd.read_csv(wd_2022, encoding='cp949')

