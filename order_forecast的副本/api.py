# -*- coding: utf-8 -*-

# from order_forecast.strategy import strategy1 as forecast_strategy
import logging
import order_forecast.strategy as ofs
import order_forecast.sim_source as ofss
# import order_forecast.config as conf
import pandas as pd
from datetime import date
from order_forecast.sim_source import session, PredictRes
from order_forecast.utils import import_conf

conf = import_conf()

def forecast1day(history_order, holidays, stg_name):
    """这里不管预测对绝对日期，给出历史数据，预测下一天对数据"""
    stg_func = ofs.__dict__[stg_name]
    forecast_periods = 1
    tomorrow_order = stg_func(history_order, holidays, forecast_periods=forecast_periods)
    return tomorrow_order

def forecast(forecast_day: str, city_id: int) -> dict:
    """预测接口, 作为服务调用
    Args:
        forecast_day: eg: '2018-01-01'
        city_id 
    Return:
        predict_res (dict):
        eg: res = {
            city_id: xxx,
            predict_day: xxx,
            predict_order_cnt: xxx,
            predict_order_upper: xxx, 
            predict_order_lower: xxx
        }
    """
    # get today date
    city_id = int(city_id)
    today = pd.Timestamp(date.today())
    forecast_day_ts = pd.Timestamp(forecast_day)

    # check forecast_day
    if forecast_day_ts < today:
        result = {'res' : -1, 'msg': "预测日期 {} 已经发生".format(forecast_day)}
        return result
    else:
        logging.debug("{} 预测时间合理".format(forecast_day))

    # check database
    today_str = today.strftime('%Y-%m-%d')
    db_res = session.query(PredictRes).filter(PredictRes.pre_date == forecast_day,
                                              PredictRes.query_date == today_str,
                                              PredictRes.city_id == city_id).all()
    # Done: 确定query_date和pre_date才行
    if len(db_res) != 0:
        logging.info("结果已存在于DB中，直接读取并返回")
        result = db_res[0]  # Done: result为SQL对象，加工成字典之后返回
        res_dict = {
            'city_id': city_id,
            'yhat': result.yhat,
            'yhat_lower': result.yhat_lower,
            'yhat_upper': result.yhat_upper,
            'forecast_day': forecast_day
        }
        logging.info(res_dict)

        return {'res': 0, 'msg': '计算结果已存在直接返回', 'values': res_dict}
    else:
        logging.info("结果不存在于BD中，现场计算")

    # calc train_date_cut
    train_periods = pd.Timedelta(conf.TRAIN_PERIODS, 'D')
    train_date_cut = today - train_periods
    logging.info("Train_date_cut: {}".format(train_date_cut))
    # get city_order
    #   - check city_order length
    city_order = ofss.get_city_order(city_id, train_date_cut)
    if len(city_order) < conf.TRAIN_MIN:
        logging.warning("city_order's length is less than {}".format(conf.TRAIN_MIN))
        result = {'res': -1, 'msg': "city {}'s history date is less than {}".format(city_id, conf.TRAIN_MIN)}
        return result

    # get holiday
    df_holiday = ofss.get_0holiday_data()

    # check holiday
    if forecast_day not in df_holiday['ds']:
        logging.warning("没有预测日期的holiday数据，可能造成预测偏差")
    df_holiday = df_holiday.dropna(axis=0)
    # use strategy to forecast
    predict_func = ofs.__dict__[conf.STG_NAME]
    # Done: predict_res改为列表，即使只有一个元素。将forecast_day到明天的日期的预测全部都记录进数据库
    predict_res = predict_func(city_order, df_holiday, forecast_day_ts)
    # predict_res (dict): keys: ds, yhat, yhat_lower, yhat_upper
    # prepare predict and save to DB
    predict_res['query_date'] = today.strftime('%Y-%m-%d')
    # db.insert(predict_res)
    all_res = []
    for row_idx, row in predict_res.iterrows():
        # push data to DB
        pre_date = row['ds'].strftime('%Y-%m-%d')
        pres = PredictRes(city_id=city_id, pre_date=pre_date, query_date=today_str, yhat=int(row['yhat']),
                          yhat_lower=int(row['yhat_lower']),
                          yhat_upper=int(row['yhat_upper']))
        all_res.append(pres)
        if row_idx == forecast_day_ts:
            pre_res = pres.get_res()

    session.add_all(all_res)
    session.commit()
    # TODO: prepare result

    print(pre_res)

    return {'res': 1, 'msg': '计算并储存计算结果', 'values': pre_res}


def main():
    res = forecast('2018-08-15', 1101)


if __name__ == '__main__':
    main()
