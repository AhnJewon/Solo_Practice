
from time import strftime
import requests
import json
import datetime

Short_weather_url = 'http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst'

serviceKey='sxYdeR9UgZrdePFHrMMqTmimWbv8FcQQzOdVAqJVlKE2MMcv6jPbI7PKOQcTdWjHr43o8i5NthaO1H06wlYWLg=='

base_date = datetime.datetime.today()
base_date = base_date.strftime('%Y%m%d')

nx=69
ny=107

params ={'serviceKey' : serviceKey,  'dataType' : 'JSON', 'base_date' : base_date, 'base_time' : '0800', 'nx' : nx , 'ny' : ny }

response = requests.get(Short_weather_url, params=params)
print(response.content)