from fbprophet import Prophet
import pandas as pd
from order_forecast.Deviation_analysis import Deviation_analysis
from termcolor import *
import datetime
import numpy as np
import matplotlib.pyplot as plt

def nfl_weekend(ds):
    date = pd.to_datetime(ds)
    if (date.weekday() == 5 or date.weekday() == 6) and date.month != 2 :
        return 1
    else:
        return 0

def load_holiday_data(holiday_data_path):
    df = pd.read_csv(holiday_data_path, parse_dates=['dt_ymd'])

    df = df[['holiday', 'dt_ymd']]
    df = df.rename(columns={ 'dt_ymd': 'ds'})
    df.loc[:, 'lower_window'] = 0
    df.loc[:, 'upper_window'] = 0

    # deal spring festval, 15days
    df.loc[df['holiday'] == '春节', 'lower_window'] = -3
    df.loc[df['holiday'] == '春节', 'upper_window'] = 7

    # deal Nation_day,8days
    df.loc[df['holiday'] == '国庆节', 'lower_window'] = 0
    df.loc[df['holiday'] == '国庆节', 'upper_window'] = 7

    # deal Worker's day
    df.loc[df['holiday'] == '劳动节', 'lower_window'] = -3
    df.loc[df['holiday'] == '劳动节', 'upper_window'] = 0

    # deal New year's day
    df.loc[df['holiday'] == '元旦', 'lower_window'] = -1
    df.loc[df['holiday'] == '元旦', 'upper_window'] = 1

    df = df.dropna()
    return df

def load_HISTORY_data(history_data_path):
    original_data = pd.read_csv(history_data_path, parse_dates=['dt_ymd'])

    city_list_default = original_data['city_id'].unique()

    return original_data,city_list_default

def get_city_data(original_data,city_list):

    city_data_train = []
    city_data_test = []

    for x in city_list:

        city =  original_data[(original_data['city_id'] == x ) & (original_data['dt_ymd'] >= '2016-6-01')].sort_values(
            by=['dt_ymd']).reset_index()

        data_train = city[['dt_ymd', '_c2','city_id']].iloc[0:int(len(city) * .9)].rename(
            columns={'dt_ymd': 'ds', "_c2": 'y'})
        data_test = city[['dt_ymd', '_c2', 'city_id']].iloc[int(len(city) * .9):len(city) - 1].rename(
            columns={'dt_ymd': 'ds', "_c2": 'y'})
        # data_train = city[['dt_ymd', '_c2', 'city_id']].iloc[0:int(len(city) * .9)].rename(columns={'dt_ymd': 'ds', "_c2": 'y'})
        # data_test = []

        city_data_train.append(data_train)
        city_data_test.append(data_test)

    return city_data_train,city_data_test

def integration(data_test, data_prediction):

    data_prediction = data_prediction[['ds', 'yhat']]

    data_for_deviation=data_test.join(data_prediction.set_index('ds'), on='ds')

    data_for_deviation = data_for_deviation.dropna()

    return data_for_deviation

def prophet(history_data_path, holiday_data_path,city_list):

    history_data, city_list_default = load_HISTORY_data(history_data_path)

    city_data_train, city_data_test = get_city_data(history_data, city_list)

    holiday = load_holiday_data(holiday_data_path)

    forecast_data = []
    future_data = []
    for x in range(len(city_list)):

        m = Prophet(holidays=holiday, yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False,
                    holidays_prior_scale=20, seasonality_mode='multiplicative')

        city_data_train[x]['nfl_weekend'] = city_data_train[x]['ds'].apply(nfl_weekend)

        m.add_regressor('nfl_weekend')

        m.fit(city_data_train[x])

        future = m.make_future_dataframe(periods=len(city_data_test[x])+20)

        future['nfl_weekend'] = future['ds'].apply(nfl_weekend)

        forecast = m.predict(future)

        if x==0:

            fig = m.plot(forecast)

            figl = m.plot_components(forecast)

        forecast_data.append(forecast)

        future_data.append(future)


    return forecast_data,city_data_train,city_data_test

def deviation(city_data_train,city_data_test,forecast_data):

    if len(city_data_test)==len(city_data_train)==len(forecast_data):
        data_for_deviation_outer = []
        data_for_deviation_inner = []
        mean_ratio_outer = []
        mean_ratio_inner = []

        for x in range(len(city_data_test)):

            data_for_deviation_outer.append(integration(city_data_test[x], forecast_data[x]))
            data_for_deviation_inner.append(integration(city_data_train[x], forecast_data[x]))

            data_for_deviation_outer[x]['deviation'] = data_for_deviation_outer[x]['y'] - data_for_deviation_outer[x]['yhat']
            data_for_deviation_outer[x]['ratio'] = abs(data_for_deviation_outer[x]['deviation'] / data_for_deviation_outer[x]['y'])

            data_for_deviation_inner[x]['deviation'] = data_for_deviation_inner[x]['y'] - data_for_deviation_inner[x]['yhat']
            data_for_deviation_inner[x]['ratio'] = abs(data_for_deviation_inner[x]['deviation'] / data_for_deviation_inner[x]['y'])

            mean_ratio_outer.append(np.mean(data_for_deviation_outer[x]['ratio']))
            mean_ratio_inner.append(np.mean(data_for_deviation_inner[x]['ratio']))

        return data_for_deviation_inner,data_for_deviation_outer,mean_ratio_inner,mean_ratio_outer
    else:
        print("输入数据维度不相同")

city_list = [1101, 3101, 5101, 3301, 4403, 4401]

forecast_data,city_data_train,city_data_test = prophet('query-hive-place.csv','holiday_deal_1.csv',city_list)

data_for_deviation_inner,data_for_deviation_outer,mean_ratio_inner,mean_ratio_outer = deviation(city_data_train,city_data_test,forecast_data)

# deviation=[]
# for i in range(len(city_list)):
#     deviation.append(Deviation_analysis(forecast_data['y'],forecast_data['yhat']))
print()

#
# fig = m.plot(forecast)
#
# figl = m.plot_components(forecast)
#
# deviation,analysis = Deviation_analysis(data_for_deviation['y'],data_for_deviation['yhat'])
# deviation1,analysis1 = Deviation_analysis(data_for_deviation1['y'],data_for_deviation1['yhat'])

# print(deviation)