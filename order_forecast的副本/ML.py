# -*- coding: utf-8 -*-
"""
ML辅助函数等
"""
import pandas as pd
import numpy as np
from pandas.tseries.offsets import DateOffset


def vector_history_order(history_order, holidays):
    """
    One-Hot and vectorify history data
    Args:
        history_order: 要求已经是单个城市对，并且按照ds排序完成之后对df，不包含city_id列，cols=[ds, y], index = ds
        current_date (pd.Timestamp): 需要使用的当前时间点
    """
    # current_date = history_order['ds'].iloc[-1]
    his_col_name = sample_rule()

    y_arr_list = []
    y_list = []
    for rowidx, row in history_order.iterrows():
        sampleidx = history_timeIndex_sample(current_date=rowidx)
        y_arr = history_order.reindex(sampleidx)['y'].values
        y_arr_list.append(y_arr)
        y_list.append(row['y'])
    y2d_arr = np.array(y_arr_list)
    df_his = pd.DataFrame(y2d_arr, columns=his_col_name,
                          index=history_order.index)

    df_his_clean = df_his.dropna(axis=0)

    #Done: 取样数据进行
    df_holiday_onehot, _ = holiday2onehot(holidays)
    # value = df_holiday_onehot.reindex(current_date).values
    df = pd.concat([df_his_clean, df_holiday_onehot], axis=1, join_axes=[df_his_clean.index])
    X = df.fillna(0)
    y = np.array(y_list)
    # vector_history_order = pd.concat([hitory_order_sampling])
    return X, y


def holiday2onehot(holidays):
    """将holiday的信息转换为Onehot数据集
    Args:
        holidays(pd.DataFram): 以ds为index，具有ds和holiday栏
    """
    onehot_cols = holidays['holiday'].unique()
    onehot_list = []
    index_list = []
    for row_index, row in holidays.iterrows():
        index_list.append(row_index)
        onehot = [1 if (row['holiday'] == ho) else 0 for ho in onehot_cols]
        onehot_list.append(onehot)

    df_onehot = pd.DataFrame(
        onehot_list, index=index_list, columns=onehot_cols)
    return df_onehot, onehot_cols


def history_timeIndex_sample(current_date):
    """退点出所需数据的历史用量
    Args：
        current_date (pd.Timestramp): 当前日期的时间戳
    """
    near60days = pd.date_range(
        end=current_date, periods=61, freq='D', closed='left').sort_values(ascending=False)

    lastyear60days = pd.timedelta_range(
        start='-30 days', end='30 days', freq='D') + current_date - DateOffset(years=1)
    lastyear60days = lastyear60days.sort_values(ascending=False)
    # lastyear60 = current_date + delta60days
    # all_index = pd.concat(near60days, lastyear60)
    # all_index = pd.DatetimeIndex([near60days, delta60days])
    all_index = near60days.append(lastyear60days)
    return all_index


def sample_rule():
    """目前先使用固定的规则"""
    col_name1 = ["-{}D".format(i) for i in range(1,61)]
    col_name2 = ["-{}D-1Y".format(i) for i in range(-30, 31)]
    col_name = col_name1 + col_name2
    return col_name
