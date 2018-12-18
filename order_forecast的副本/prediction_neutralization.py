from fbprophet import Prophet
import pandas as pd
from termcolor import *
import datetime
import numpy as np

def neutralization(forecast1,forecast2,forecast3,forecast4,date,citylist,weight1=.38,weight2=.38,weight3=.12,weight4=.12):

    if len(forecast1)==len(forecast2)==len(forecast3)==len(forecast4)==len(citylist):

        date = '%d-%d-%d' % (date[0], date[1], date[2])

        prediction_inf_box=[]

        city_number=len(citylist)

        for x in range(city_number):

            prediction1 = forecast1[x]['yhat'].loc[forecast1[x]['ds'] == date].values[0]
            prediction2 = forecast2[x]['yhat'].loc[forecast2[x]['ds'] == date].values[0]
            prediction3 = forecast3[x]['yhat'].loc[forecast3[x]['ds'] == date].values[0]
            prediction4 = forecast4[x]['yhat'].loc[forecast4[x]['ds'] == date].values[0]

            prediction = prediction1 * weight1 + prediction2 * weight2 + prediction3 * weight3 + prediction4 * weight4

            city = citylist[x]

            prediction_inf = [city,date,prediction]

            prediction_inf_box.append(prediction_inf)

        return prediction_inf_box

    else:

        print('预测数据规模互不匹配，错误发生在prediction_neutralization')

        return []
