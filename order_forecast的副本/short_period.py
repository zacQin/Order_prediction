from fbprophet import Prophet
import pandas as pd
from order_forecast.Deviation_analysis import Deviation_analysis
from termcolor import *
import datetime
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.diagnostic import  acorr_ljungbox
import matplotlib.pyplot as plt

def load_HISTORY_data(history_data_path):
    original_data = pd.read_csv(history_data_path, parse_dates=['dt_ymd'])
    # city_list = original_data['city_id'].unique()
    peking = original_data[(original_data['city_id']==1101)&(original_data['dt_ymd']>='2018-8-09')&(original_data['dt_ymd']<='2018-9-09')].sort_values(by=['dt_ymd']).reset_index()
    #& (original_data['dt_ymd']>='2016-01-01')  (original_data['city_id']==5101)

    # data_train = peking[['dt_ymd','_c2']].iloc[0:int(len(peking)*.95)].rename(columns={'dt_ymd':'ds',"_c2":'y'})
    # data_test = peking[['dt_ymd','_c2']].iloc[int(len(peking)*.95):len(peking)].rename(columns={'dt_ymd':'ds',"_c2":'y'})

    data_train = peking[['dt_ymd', '_c2']].iloc[0:int(len(peking))].rename(columns={'dt_ymd': 'ds', "_c2": 'y'})
    data_test=[]

    return original_data,data_train,data_test

original_data,data_train,data_test = load_HISTORY_data('query-hive-155230.csv')
# holiday = load_holiday_data('holiday.csv')

m=Prophet( yearly_seasonality=False, weekly_seasonality=True, daily_seasonality=False,seasonality_mode='multiplicative')

m.fit(data_train)

future = m.make_future_dataframe(periods=10)

forecast = m.predict(future)

fig = m.plot(forecast)

figl = m.plot_components(forecast)

print('')