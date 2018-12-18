from fbprophet import Prophet
import pandas as pd
from order_forecast.Deviation_analysis import Deviation_analysis
from termcolor import *
import datetime
import numpy as np
import matplotlib.pyplot as plt
from fbprophet.plot import add_changepoints_to_plot  #展示转折点
from fbprophet.plot import plot_yearly  #展示年变化过程

df=pd.read_csv('example_wp_log_R.csv')
#
# #设置假期模块
holiday=pd.DataFrame({'holiday':'National Day','ds':pd.to_datetime(['2008-10-1','2009-10-1']),'lower_windows':-3,'upper_windows':5})
# #假日带来的效果可以通过foreca查看
#
# #建立类对象 holidays=holiday,changepoints=[],,changepoint_prior_scale=.1,yearly_seasonality=20/False
m=Prophet(n_changepoints=1,changepoint_range=0.9,changepoint_prior_scale=.1,changepoints=[],holidays=holiday,yearly_seasonality=20/False,holidays_prior_scale=0.05,seasonality_mode='multiplicative')
#          #预设转折点数量     设定转折点可存在的范围   转折点拟合灵活度，越高越灵活   指定转折点位置    加入假期          年拟合灵敏度                  假期拟合灵活度
#
#          #m.add_seasonality(name='monthly',period=30.5,fourier_order=5，prior_scale)
#          #增加季节性规律                     预测数量     傅里叶级数        灵活度
#
#          #m.add_regressor('')   增加额外回归量
m.fit(df) #拟合数据
#
future=m.make_future_dataframe(periods=365) #扩展数据容量
#
forecast=m.predict(future) #预测
#
fig=m.plot(forecast) #展示
#        # a=add_changepoints_to_plot(fig.gca(),m,forecast) #展示转折点
fig=m.plot_components(forecast) #各成分展示

print('')