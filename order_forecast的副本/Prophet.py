from fbprophet import Prophet
import pandas as pd
# from order_forecast.Deviation_analysis import Deviation_analysis
from termcolor import *
import datetime
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.diagnostic import  acorr_ljungbox
import matplotlib.pyplot as plt

def load_holiday_data(holiday_data_path):
    df = pd.read_csv(holiday_data_path, parse_dates=['cd_holiday_ymd.dt_ymd'])

    df = df[['cd_holiday_ymd.festival', 'cd_holiday_ymd.dt_ymd']]
    df = df.rename(columns={'cd_holiday_ymd.festival': 'holiday', 'cd_holiday_ymd.dt_ymd': 'ds'})
    df.loc[:, 'lower_window'] = 0
    df.loc[:, 'upper_window'] = 0

    # # deal New year's day
    # df.loc[df['holiday'] == '元旦', 'lower_window'] = -1
    # df.loc[df['holiday'] == '元旦', 'upper_window'] = 1
    #
    # # deal spring festval, 15days
    # df.loc[df['holiday'] == '春节', 'lower_window'] = -3
    # df.loc[df['holiday'] == '春节', 'upper_window'] = 8
    #
    # #deal 清明节
    # df.loc[df['holiday'] == '清明', 'lower_window'] = -2
    # df.loc[df['holiday'] == '清明', 'upper_window'] = 2
    #
    # # deal Worker's day
    # df.loc[df['holiday'] == '劳动节', 'lower_window'] = -2
    # df.loc[df['holiday'] == '劳动节', 'upper_window'] = 0
    #
    # #deal 端午
    # df.loc[df['holiday'] == '端午节', 'lower_window'] = -2
    # df.loc[df['holiday'] == '端午节', 'upper_window'] = 1
    #
    # #deal 中秋
    # df.loc[df['holiday'] == '中秋节', 'lower_window'] = -2
    # df.loc[df['holiday'] == '中秋节', 'upper_window'] = 1
    #
    # # deal Nation_day,8days
    # df.loc[df['holiday'] == '国庆节', 'lower_window'] = 0
    # df.loc[df['holiday'] == '国庆节', 'upper_window'] = 7
    #
    # df = df.dropna()

    return df

# clean original_data
def load_HISTORY_data(history_data_path):
    original_data = pd.read_csv(history_data_path, parse_dates=['dt_ymd'])
    # city_list = original_data['city_id'].unique()
    peking = original_data[(original_data['city_id']==4401)&(original_data['dt_ymd']>='2017-1-01')&(original_data['dt_ymd']<='2018-9-09')].sort_values(by=['dt_ymd']).reset_index()
    #& (original_data['dt_ymd']>='2016-01-01')  (original_data['city_id']==5101)

    data_train = peking[['dt_ymd','_c2']].iloc[0:int(len(peking)*.9)].rename(columns={'dt_ymd':'ds',"_c2":'y'})
    data_test = peking[['dt_ymd','_c2']].iloc[int(len(peking)*.9):len(peking)].rename(columns={'dt_ymd':'ds',"_c2":'y'})

    # data_train = peking[['dt_ymd', '_c2']].iloc[0:int(len(peking))].rename(columns={'dt_ymd': 'ds', "_c2": 'y'})
    # data_test=[]

    return original_data,data_train,data_test

def integration(data_test, data_prediction):

    data_prediction = data_prediction[['ds', 'yhat']].iloc[0:-1]

    data_for_deviation=data_test.join(data_prediction.set_index('ds'), on='ds')

    data_for_deviation = data_for_deviation.dropna()

    return data_for_deviation

def nfl_friday(ds):
    date = pd.to_datetime(ds)

    if (date.weekday() == 5 or date.weekday() == 6)  :
        #and date.month !=
        return 1
    else:
        return 0


#builde model
original_data,data_train,data_test = load_HISTORY_data('query-hive-155230.csv')
# holiday = load_holiday_data('holiday.csv')
holiday = pd.read_csv('holiday_deal_1.csv',parse_dates=['dt_ymd'])
holiday = holiday.loc[:, ['dt_ymd', 'holiday']].rename(columns={'dt_ymd':'ds'})
holiday.loc[:, 'lower_window'] = 0
holiday.loc[:, 'upper_window'] = 0
holiday.loc[holiday['holiday'] == '除夕', 'lower_window'] = -3
holiday.loc[holiday['holiday'] == '春节', 'upper_window'] = 5
holiday.loc[holiday['holiday'] == '春节收假', 'upper_window'] = 2
holiday.loc[holiday['holiday'] == '国庆节', 'upper_window'] = 6
# holidays_prior_scale=20,seasonality_mode='multiplicative',holidays_prior_scale=5
m=Prophet(holidays=holiday, yearly_seasonality=True, weekly_seasonality=True,daily_seasonality=False,holidays_prior_scale=35,seasonality_mode='multiplicative')

data_train['nfl_friday'] = data_train['ds'].apply(nfl_friday)
m.add_regressor('nfl_friday')
m.fit(data_train)


future = m.make_future_dataframe(periods=len(data_test)+10)
future['nfl_friday'] = future['ds'].apply(nfl_friday)

# for x in holiday['ds'][holiday['holiday']=='加班']:
#     future['nfl_friday'][future['ds'] == x ] = 0

forecast = m.predict(future)

# forecast['yhat'][forecast['yhat'] <0] = 5000


# fig = \
m.plot(forecast)

figl = m.plot_components(forecast)

def analysis(data_test, data_train, forecast):
    data_for_deviation = integration(data_test, forecast)
    data_for_deviation1 = integration(data_train, forecast)

    # deviation = Deviation_analysis(data_for_deviation['y'], data_for_deviation['yhat'])
    # deviation1 = Deviation_analysis(data_for_deviation1['y'], data_for_deviation1['yhat'])

    data_for_deviation['deviation'] = data_for_deviation['yhat'] - data_for_deviation['y']
    data_for_deviation['ratio'] = abs(data_for_deviation['deviation'] / data_for_deviation['y'])
    data_for_deviation1['deviation'] = data_for_deviation1['yhat'] - data_for_deviation1['y']
    data_for_deviation1['ratio'] = abs(data_for_deviation1['deviation'] / data_for_deviation1['y'])

    mean_ratio = np.mean(data_for_deviation['ratio'])
    mean_ratio1 = np.mean(data_for_deviation1['ratio'])

    fore = forecast.loc[:, ['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

    diff0 = pd.concat([data_for_deviation1['deviation'],data_for_deviation['deviation']])
    diff1 = pd.concat([data_for_deviation1['ds'],data_for_deviation['ds']])
    diff = pd.concat([diff1,diff0],axis=1).reset_index().reset_index()
    # print('')
    for xx in range(7):
        sum_ratio = 0
        num = 0
        for x in range(len(data_for_deviation)-1):
            if x % 7==xx:
                ratio = data_for_deviation['ratio'].iloc[x]
                num+=1
                sum_ratio += ratio
        finaly_ratio = sum_ratio/num
        print(xx,'zhi',finaly_ratio)


    return diff


diff = analysis(data_test, data_train ,forecast)
#
# plt.figure()
# plt.scatter(diff['level_0'],diff['deviation'],s=.1)
# plt.figure()

# plt.plot(diff['ds'],diff['deviation'], marker='o', mec='b', mfc='w')

# x = holiday.loc['ds'][holiday['ds']>='2016-6-01'].values
# diff.to_csv('diff.csv')
# holiday.to_csv('holiday_deal.csv')

# for ii in range(len(x)):
# plt.axvline('20170828',color='r',alpha=.2)
# # plt.show()
#
# m=Prophet(seasonality_prior_scale=60)
# ss=diff.loc[:,['ds','deviation']].rename(columns={'deviation':'y'})
# m.fit(ss)
#
# future=m.make_future_dataframe(periods=50)
# forecast = m.predict(future)
# fig = m.plot(forecast)
# figl = m.plot_components(forecast)
#
# dta=diff[['deviation']].values
#
# noiseres=acorr_ljungbox(dta,lags=1)
# print (u'一阶差分序列的白噪声检验结果为：')
# print ('stat                  | p-value')
# for x in noiseres:
#     print(x,'|')
#


print('')

# data_for_difference = pd.concat(forecast.loc['ds','yhat'],np.vstack(data_train['y'],data_test['y']))
