# -*- coding: utf-8 -*-
"""不同的预测策略"""
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fbprophet import Prophet


def strategy1(history_order, holidays, forecast_periods=1):
    """
    使用FBprophet模块进行一天的价格预测
    原来的df都是用的date作为index，这里需要reset index
    """
    # Done: reset_index() and rename TimeIndex name to 'ds

    # 仅仅使用60天的数据，进行判断
    history_len = 30
    if history_order.shape[0] < history_len:
        raise ValueError(
            'history data length is not enough, need at least {} data'.format(history_len))
    # 使用累加性数据，使用对数化数值
    log_y = np.log(history_order['y'].values)
    history_order.loc[:, 'y'] = log_y

    # Done: check holiday datetime

    model = Prophet(holidays=holidays)
    model.fit(history_order)
    future = model.make_future_dataframe(periods=1)
    # BUG: 这里有个问题，future中没有带着holiday信息，怎么预测？
    predict = model.predict(future)
    res = predict.iloc[-1]
    res = res[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    res[['yhat', 'yhat_lower', 'yhat_upper']] = np.exp(
        res[['yhat', 'yhat_lower', 'yhat_upper']].astype(np.float32))
    # res = np.exp(res)
    return res


def strategy2(history_order, holidays, forecast_periods=1):
    """不使用log的累加性FBProphet模型"""
    model = Prophet(holidays=holidays, yearly_seasonality=False,
                    weekly_seasonality=True)
    model.fit(history_order)
    # model.fit()
    future = model.make_future_dataframe(periods=1)
    predict = model.predict(future)
    predict = predict.iloc[-1]
    res = predict[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    return res


def strategy_qixi(city_order, holiday, forecast_day, forecast_periods=14):
    model = Prophet(holidays=holiday, yearly_seasonality=True,
                    weekly_seasonality=True, seasonality_mode='multiplicative')
    model.fit(city_order)
    future = model.make_future_dataframe(periods=14)
    predict = model.predict(future)

    # res = predict.loc[forecast_day, ['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    predict2 = predict.set_index('ds', drop=False)
    res2 = predict2.loc['2018-08-16':'2018-08-17',
           ['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    return res2


def strategy_qixi_add(city_order, holiday, fname, forecast_periods=14):
    """累加性模型"""
    model = Prophet(holidays=holiday, yearly_seasonality=True,
                    weekly_seasonality=True)
    model.fit(city_order)
    future = model.make_future_dataframe(periods=forecast_periods)
    predict = model.predict(future)

    predict2 = predict.set_index('ds', drop=False)
    res2 = predict2.loc['2018-08-16':'2018-08-17',
           ['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

    _ = plt.figure(figsize=(8, 6))
    model.plot(predict)
    plt.savefig(fname)
    print('{} saved!'.format(fname))
    return res2


def strategy_api(order_city, df_holiday, forecast_day: pd.Timestamp ):
    """作为给API调用的接口，使用FBPropht的累加性模型

    """
    model = Prophet(holidays=df_holiday, yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    model.fit(order_city)
    # future =
    # calc the forecast_periods from forecast_day
    last_day = order_city.index[-1]

    delta_t = forecast_day - last_day
    logging.info("预测未来时长是: {}".format(delta_t))
    forecast_periods = delta_t.days  # 这里不用 + 1， 预测时间精确
    future = model.make_future_dataframe(periods=forecast_periods)
    predict = model.predict(future)
    predict2 = predict.set_index('ds', drop=False)
    predict2.index.name = None
    # Done: 这里改成将所有的日期都返回
    future_cut = last_day + pd.Timedelta('1 days')
    res = predict2.loc[future_cut:, ['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
    return res


def strategy3(history_order, holidays, forecast_periods=1):
    """使用机器学习和广义线性模型"""
