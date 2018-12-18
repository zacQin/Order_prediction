from fbprophet import Prophet
import pandas as pd
from order_forecast.Deviation_analysis import Deviation_analysis
from termcolor import *
import datetime
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv('query-hive-place.csv', parse_dates=['dt_ymd'])
df = df.loc[:,["dt_ymd",'_c2']]


df_city = df.groupby('dt_ymd', as_index=False).sum()
df_city = df_city.set_index('dt_ymd', drop=False)
df_city.index.name = None
df_city.sort_index(inplace=True)
df_city.reset_index(drop=True)



print(df_city.shape)
rr=pd.DataFrame({'dt_ymd':df_city['dt_ymd'],'_c2':df_city['_c2']})
rr=rr.reset_index(drop=True)
rr.to_csv('country_data.csv')



df_city = df['_c2'].groupby(df['dt_ymd']).sum()
# dff=df_city.sum()
# ff=pd.DataFrame(dff)
# day_data=pd.DataFrame({'dt_ymd':df['dt_ymd'].sort_values,'order_numn':sum_day.values})
sum_day = pd.concat([sum_day,df['dt_ymd']])
# sum_day = sum_day.set_index('dt_ymd', drop=False)
# df_city.index.name = None
s=df_city.loc[:,['dt_ymd','Unnamed: 0']].rename(columns={ "Unnamed: 0": '_c2'})
# df_city.sort_index(inplace=True)
ss=df_city['Unnamed: 0'] - s['_c2']

s.to_csv('country_data.csv')
print('')

print()