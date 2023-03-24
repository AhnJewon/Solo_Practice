import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split

weather = 'https://github.com/AhnJewon/Solo_Practice/blob/35f8fc3c910b382dc079a9838b3330cc34a141ea/OBS_ASOS_TIM_20230324165804.csv'

chungju = pd.read_csv(weather)
print(chungju.columns)
chungju.head()

tem = chungju[['기온(°C)']]

print(tem)
