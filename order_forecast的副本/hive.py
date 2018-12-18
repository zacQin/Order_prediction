#!/usr/bin/env python3
# _*_ coding: utf-8 _*_
"""查询并下载hive数据的接口"""
from pyhive import hive
import pandas as pd

hostname = '10.81.11.189'


# def selectFromHive(select_sql):
#     conn = hive.connect(hostname)
#     cur = conn.cursor()
#
#     cur.execute(select_sql)
#     rows = cur.fetchall()
#
#     cur.close()
#     conn.close()
#     return rows
def get_data(sql_str):
    conn = hive.connect(hostname)
    df = pd.read_sql(sql=sql_str, con=conn)
    return df


if __name__ == '__main__':
    select_sql = 'SELECT dt_ymd,city_id, sum(place_cnt) FROM edw.md_order_user_ymd GROUP BY dt_ymd, city_id'
    df = get_data(select_sql)
    df.to_csv('/home/gaopuyuan/data/query-hive-place.csv')
    print(df.head())
