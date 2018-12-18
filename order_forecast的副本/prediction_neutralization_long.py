from fbprophet import Prophet
import pandas as pd
from termcolor import *
import datetime
import time
import numpy as np

def neutralization_long(forecast1,forecast2,forecast3,forecast4,citylist,weight1=.4,weight2=.4,weight3=.1,weight4=.1):

    if len(forecast1)==len(forecast2)==len(forecast3)==len(forecast4)==len(citylist):

        # date = '%d-%d-%d' % (date[0], date[1], date[2])
        #
        prediction_inf_box=[]

        prediction_inf_box_ign=[]

        city_number=len(citylist)

        for x in range(city_number):

            prediction1 = forecast1[x]
            prediction2 = forecast2[x]
            prediction3 = forecast3[x]
            prediction4 = forecast4[x]

            prediction1 = prediction1.loc[:,['ds','yhat','yhat_upper','yhat_lower']].rename(columns={'yhat':'yhat_1'})
            prediction2 = prediction2.loc[:,['ds','yhat']].rename(columns={'yhat':'yhat_2'})
            prediction3 = prediction3.loc[:,['ds','yhat']].rename(columns={'yhat':'yhat_3'})
            prediction4 = prediction4.loc[:,['ds','yhat']].rename(columns={'yhat':'yhat_4'})

            prediction = pd.merge(prediction1,prediction2,on='ds')
            prediction = pd.merge(prediction,prediction3,on='ds')
            prediction = pd.merge(prediction,prediction4,on='ds')

            date = time.localtime(time.time())

            date = '%d-%d-%d' % (date[0], date[1], date[2])

            prediction['calc_date'] = date

            prediction['city'] = citylist[x]

            prediction_igno_short = prediction

            prediction['yhat']= (prediction['yhat_1'] * weight1) + (prediction['yhat_2'] * weight2) + (prediction['yhat_3'] * weight3) + (prediction['yhat_4'] * weight4)

            prediction['yhat_lower'] = prediction['yhat_lower'] + (prediction['yhat'] - prediction['yhat_1'])

            prediction['yhat_upper'] = prediction['yhat_upper'] + (prediction['yhat'] - prediction['yhat_1'])

            prediction.drop(['yhat_1','yhat_2','yhat_3','yhat_4'],axis=1,inplace=True)

            del prediction['yhat_1']
            del prediction['yhat_2']
            del prediction['yhat_3']
            del prediction['yhat_4']

            prediction_inf_box.append(prediction)

            # prediction_igno_short['yhat'] = (prediction['yhat_1'] * .5) + (prediction['yhat_2'] * .5)

            # prediction_inf_box.append(prediction.loc[:,['ds','city','yhat']])

            # prediction_inf_box_ign.append(prediction_igno_short.loc[:, ['ds', 'city', 'yhat']])


        return prediction_inf_box

    else:

        print('预测数据规模互不匹配，错误发生在prediction_neutralization')

        return []
