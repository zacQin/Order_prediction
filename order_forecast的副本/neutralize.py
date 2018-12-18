import pandas as pd
import math
import numpy as np
beijing1 = pd.read_csv('beijing_1.csv').loc[:,['ds','yhat']]
beijing2 = pd.read_csv('beijing_2.csv').loc[:,['ds','yhat']]
beijing3 = pd.read_csv('beijing_3.csv').loc[:,['ds','yhat']]
beijing4 = pd.read_csv('beijing_4.csv').loc[:,['ds','yhat__']]

beijing_data = pd.read_csv("beijing_data").rename(columns={'dt_ymd':'ds'})
beijing = pd.merge(beijing1,beijing2,on='ds')
beijing = pd.merge(beijing,beijing3,on='ds')
beijing = pd.merge(beijing,beijing4,on='ds')

beijing = pd.merge(beijing_data.loc[:,['ds','_c2']],beijing)

beijing['yhat_final'] = .8*beijing['yhat_x']+.1*beijing['yhat']+.1*beijing['yhat__']
beijing['yhat_final__'] = .39*beijing['yhat_x']+.39*beijing['yhat_y']+.11*beijing['yhat__']+.11*beijing['yhat']

# beijing['yhat_final'] = .6*beijing['yhat_x']+.4*beijing['yhat']

beijing['deviation'] = beijing['yhat_final']-beijing['_c2']
beijing['ratio'] = abs(beijing['deviation']/beijing['_c2'])

beijing['deviation__'] = beijing['yhat_final__']-beijing['_c2']
beijing['ratio__'] = abs(beijing['deviation__']/beijing['_c2'])

beijing_test = beijing.iloc[math.ceil(len(beijing)*.9):]
beijing_train = beijing.iloc[0:math.ceil(len(beijing)*.9)]

ratio_train = np.mean(beijing_train['ratio'])
ratio_test = np.mean(beijing_test['ratio'])

ratio_train__ = np.mean(beijing_train['ratio__'])
ratio_test__ = np.mean(beijing_test['ratio__'])

print()