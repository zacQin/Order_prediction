import numpy as np
from termcolor import *
import math
import numpy as np
import pandas as pd

def Deviation_analysis(original_data, prediction_data):

    if len(original_data) == len(prediction_data):
        days = len(prediction_data)
    else:
        print(colored('original_data should be the same size as prediction_data',"red"))
        return 'error'

    deviation_array = (prediction_data - original_data)

    ### calculate MSE ###
    MSE = sum(deviation_array**2) / days
    # print(deviation_array, deviation_array**2, sum(deviation_array**2), days)

    ### calculate MAE ###
    MAE = sum(abs(deviation_array)) / days
    # print(deviation_array, abs(deviation_array), sum(abs(deviation_array)), days)

    ### calculate RMSE ###
    RMSE = math.sqrt(MSE)

    ### calculate RMAE ###
    RMAE = math.sqrt(MAE)

    ### calculate Log_cosh ###
        #cosh=(exp(x)+exp(-x))/2
        #Log_Cosh = sum(math.log((math.exp(x) + math.exp(-x)) * .5) for x in deviation_array)
    #Log_Cosh = np.sum(np.log(np.cosh(deviation_array)))

    ### calculate Huber_loss ###
    ### 未完成
    # loss = np.where(np.abs(deviation_array) < delta, 0.5 * ((deviation_array) ** 2), delta * np.abs(deviation_array) - 0.5 * (delta ** 2))

    result=pd.DataFrame({"MSE":[MSE],"MAE":[MAE],"RMSE":[RMSE],"RMAE":[RMAE]})
    #"Log_cosh":[Log_Cosh]
    return result