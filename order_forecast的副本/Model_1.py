from fbprophet import Prophet
import pandas as pd
from termcolor import *
import datetime
import numpy as np
from pretreatment import *

#预测模型
def prophet_1(city_data_train, holiday, city_list,period):

    forecast_data = []
    future_data = []

    for x in range(len(city_list)):

        m = Prophet(holidays=holiday, yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False,
                    holidays_prior_scale=35, seasonality_mode='multiplicative')

        city_data_train[x]['nfl_weekend'] = city_data_train[x]['ds'].apply(nfl_weekend)

        m.add_regressor('nfl_weekend')

        m.fit(city_data_train[x])

        future = m.make_future_dataframe(periods=period)

        future['nfl_weekend'] = future['ds'].apply(nfl_weekend)

        forecast = m.predict(future)



        m.plot(forecast)

        figl = m.plot_components(forecast)



        forecast_data.append(forecast)

        future_data.append(future)

    return forecast_data

# city_list = [1101, 3101, 5101, 3301, 4403, 4401]

# forecast_data,city_data_train,city_data_test = prophet('query-hive-place.csv','holiday_deal_1.csv',city_list)

# print()
