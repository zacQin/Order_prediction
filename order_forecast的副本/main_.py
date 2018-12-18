from Model_1 import prophet_1
from Model_2 import prophet_2
from Model_3 import prophet_3
from Model_4 import prophet_4
from pretreatment import *
from prediction_neutralization import *
from prediction_neutralization_long import *
from entrance import *
import time


city = [1101,3301]

result = prediction_entrance(period=7,history_data_path='query-hive-20180916.csv',city_list= [1101,3301])

 # neutralization(,date=time.localtime(time.time()),citylist=city_list)

print()
    # print(prediction)