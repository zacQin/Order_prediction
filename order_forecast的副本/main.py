# -*- coding: utf-8 -*-
"""本地运行主程序"""

import datetime
import pandas as pd
from functools import partial
from multiprocessing.pool import Pool
import order_forecast.api as ofapi
import order_forecast.sim_source as ss
import order_forecast.strategy as ofs

import logging
logging.basicConfig(level='DEBUG')
from order_forecast.utils import import_conf

conf = import_conf()

history_data_path = './data/history_order.csv'
history_data_path2 = './data/query-hive-2018-08-08.csv'
history_data_path3 = './data/query-hive-place.csv'
holiday_data_path = './data/holiday.csv'
save_path = './data/pre_{}.csv'.format(
    datetime.datetime.now().strftime("%Y.%m.%d-%H:%M:%S"))


def main():
    df_his_order, city_id_list = ss.load_order_source(history_data_path)
    df_holiday = ss.load_holiday_data(holiday_data_path)

    # 使用数据：
    # - 训练：
    #     - 开始：'2018-05-15'
    #     - 结束：'2018-07-15'
    #     - 预测：'2018-07-16'
    # - 测试：
    #     - 对照：'2018-07-16'

    train_start_date = '2018-06-15'
    train_end_date = '2018-07-15'
    test_start_date = '2018-07-16'
    test_end_date = '2018-07-17'
    stg_name = 'strategy2'

    cutoff = '2018-07-15'
    train_periods = 60
    validate_periods = 1

    predict_res = []

    for city_id in city_id_list:  # 测试
        # df_city = df_his_order[[df_his_order['city_id'] == city_id]]
        # 仅仅使用60日数据进行测试
        df_city_train, df_city_validation = ss.get_local_data(
            df_his_order, city_id, cutoff, train_periods, validate_periods)
        if df_city_train['ds'].hasnans:
            print('city: {} MISSING Data on some datetime'.format(city_id))
            continue
        predict_order = ofapi.forecast1day(df_city_train, df_holiday, stg_name)
        if (predict_order['ds'] != df_city_validation['ds']).all():
            raise ValueError('预测日期和验证对照日期不匹配')
        else:
            # predict_order['y'] = df_city_validation[predict_order['ds'] == df_city_validation['ds']]['y'].values[0]
            # predict_res[city_id] = predict_order
            pre_dict = {}
            pre_dict['y'] = df_city_validation[predict_order['ds']
                                               == df_city_validation['ds']]['y'].values[0]
            pre_dict['yhat'] = float(predict_order['yhat'])
            pre_dict['yhat_lower'] = float(predict_order['yhat_lower'])
            pre_dict['yhat_upper'] = float(predict_order['yhat_upper'])
            pre_dict['city_id'] = city_id
            predict_res.append(pre_dict)
    ss.save_predict(predict_res, save_path)


def predict_qixi():
    train_start_date = '2016-08-01'
    train_end_date = 'now'
    forecast_day = '2018-08-17'
    df_order, city_list = ss.load_order_source(history_data_path2)
    df_holiday = ss.load_holiday_data(holiday_data_path)
    # predict_dict = {}
    predict_list = []
    for city_id in city_list:
        print("predict city {}".format(city_id))
        city_order = df_order[df_order['city_id'] == city_id].sort_index()
        city_order_cut = city_order[city_order['ds'] >= train_start_date]
        if len(city_order_cut) < 60:
            print("city {} data len is < 60, skipped".format(city_id))
            continue
        res = ofs.strategy_qixi(city_order_cut, holiday=df_holiday, forecast_day=forecast_day)
        print(res)
        for row_idx, row in res.iterrows():
            pre_dict = {}
            pre_dict['ds'] = row_idx
            pre_dict['yhat'] = float(row['yhat'])
            pre_dict['yhat_lower'] = float(row['yhat_lower'])
            pre_dict['yhat_upper'] = float(row['yhat_upper'])
            pre_dict['city_id'] = city_id
            predict_list.append(pre_dict)
    ss.save_predict(predict_list, save_path)


def predict_qixi_add():
    train_start_date = '2016-08-01'
    df_order, city_list = ss.load_order_source(history_data_path2)
    df_holiday = ss.load_holiday_data(holiday_data_path)
    city_order_list = []

    city_list2 = [1101, 3101, 4403]  # 北京，上海， 深圳，

    # 全国
    all_order = df_order.groupby('ds').sum()  # 全国
    all_order.reset_index(inplace=True)
    all_order = all_order.set_index('ds', drop=False)
    all_order.index.name = None
    all_order_cut = all_order[all_order['ds'] >= train_start_date]
    city_order_list.append((all_order_cut, 0, './data/fig/qixi_0.png'))

    for city_id in city_list2:
        print("predict city {}".format(city_id))
        city_order = df_order[df_order['city_id'] == city_id].sort_index()
        city_order_cut = city_order[city_order['ds'] >= train_start_date]
        if len(city_order_cut) < 360:
            print("city {} data len is < 60, skipped".format(city_id))
            continue
        else:
            figname = './data/fig/qixi_{}.png'.format(int(city_id))
            city_order_list.append((city_order_cut, city_id, figname))

    predict_qixi_func = partial(ofapi.forecast_qixi, holidays=df_holiday)

    with Pool() as p:
        predict_res = p.map(predict_qixi_func, city_order_list)

    predict_list = []
    for pre_i in predict_res:
        predict_list.append(pre_i[0])
        predict_list.append(pre_i[1])
    ss.save_predict(predict_list, save_path)


def calc_all_city():
    """并行计算所有城市的预测值"""
    city_list = ss.get_city_list(conf.city_id_path)
    today = pd.Timestamp(datetime.datetime.today())
    forecastday = today + pd.Timedelta(14, 'D')
    forecastday_str = forecastday.strftime("%Y-%m-%d")
    func_args = [(forecastday_str, int(city_id_i)) for city_id_i in city_list]
    # predict_func = partial(ofapi.forecast, forecast_day=forecastday_str)

    with Pool(processes=10) as p:
        # predict_res = p.map(predict_func, city_list)
        predict_res = p.starmap(ofapi.forecast, func_args)


def sim_call_api():
    # order_city = 1101
    order_city = 1101  # country
    forecast_day = '2018-08-15'
    res = ofapi.forecast(forecast_day, city_id=order_city)
    return res


def test1():
    print('test1')


if __name__ == '__main__':
    # main()
    # predict_qixi_add()
    # sim_res = sim_call_api()
    # sim_call_api()
    calc_all_city()