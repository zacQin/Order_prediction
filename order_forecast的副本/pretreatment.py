from fbprophet import Prophet
import pandas as pd
from termcolor import *
import datetime
import numpy as np
import matplotlib.pyplot as plt
import time

def nfl_weekend(ds):
    date = pd.to_datetime(ds)
    if (date.weekday() == 5 or date.weekday() == 6):
        return 1
    else:
        return 0

#加载节假日信息
def load_holiday_data(holiday_data_path):
    holiday = pd.read_csv(holiday_data_path, parse_dates=['dt_ymd'])

    holiday = holiday[['festival', 'dt_ymd']]
    holiday = holiday.rename(columns={'festival':'holiday','dt_ymd': 'ds'})

    holiday.loc[:, 'lower_window'] = 0
    holiday.loc[:, 'upper_window'] = 0

    #春节假前订单下降缓冲区
    holiday.loc[holiday['holiday'] == '除夕', 'lower_window'] = -3

    holiday.loc[holiday['holiday'] == '春节', 'upper_window'] = 5

    #春节假后订单回升缓冲区
    holiday.loc[holiday['holiday'] == '春节收假', 'upper_window'] = 2

    holiday.loc[holiday['holiday'] == '国庆节', 'upper_window'] = 6

    # holiday.loc[holiday['holiday'] == '中秋节', 'lower_window'] = -3

    return holiday

#加载历史订单信息
def load_HISTORY_data(history_data_path):
    original_data = pd.read_csv(history_data_path, parse_dates=['dt_ymd'])



    city_list_default = original_data['city_id'].unique()

    return original_data,city_list_default

def Model_1_date_selection(original_data):

    start_date = time.localtime(time.time() - 365 * 1.5 * 24 * 60 * 60)

    date = '%d-%d-%d' % (start_date[0], start_date[1], start_date[2])

    original_data = original_data[original_data['dt_ymd'] >= date]

    return original_data

def Model_2_date_selection(original_data):

    start_date = time.localtime(time.time() - 365 * 2.5 * 24 * 60 * 60)

    date = '%d-%d-%d' % (start_date[0], start_date[1], start_date[2])

    original_data = original_data[original_data['dt_ymd'] >= date]

    return original_data

def Model_3_date_selection(original_data):

    start_date = time.localtime(time.time() - 365 * 2 * 24 * 60 * 60)

    date = '%d-%d-%d' % (start_date[0], start_date[1], start_date[2])

    original_data = original_data[original_data['dt_ymd'] >= date]

    return original_data

def Model_4_date_selection(original_data):

    start_date = time.localtime(time.time() - 40 * 24 * 60 * 60)

    date = '%d-%d-%d' % (start_date[0], start_date[1], start_date[2])

    original_data = original_data[original_data['dt_ymd'] >= date]

    return original_data

#历史订单数据分类（按城市）
def get_city_data(original_data,city_list):

    city_data_train = []

    for x in city_list:

        city =  original_data[(original_data['city_id'] == x )].sort_values( by=['dt_ymd']).reset_index()

        data_train = city[['dt_ymd', '_c2','city_id']].rename(columns={'dt_ymd': 'ds', "_c2": 'y'})

        city_data_train.append(data_train)

    return city_data_train
