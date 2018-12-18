from Model_1 import prophet_1
from Model_2 import prophet_2
from Model_3 import prophet_3
from Model_4 import prophet_4
from pretreatment import *
from prediction_neutralization import *
from prediction_neutralization_long import *
import time

# if __name__ == '__main__':

city_list = [1101]

holiday = load_holiday_data('holiday_deal_1.csv')

original_data,city_list_default = load_HISTORY_data('query-hive-20180916_xiao.csv')

data_for_model_1 = Model_1_date_selection(original_data)
data_for_model_2 = Model_2_date_selection(original_data)
data_for_model_3 = Model_3_date_selection(original_data)
data_for_model_4 = Model_4_date_selection(original_data)

data_for_model_1 = get_city_data(data_for_model_1,city_list)
data_for_model_2 = get_city_data(data_for_model_2,city_list)
data_for_model_3 = get_city_data(data_for_model_3,city_list)
data_for_model_4 = get_city_data(data_for_model_4,city_list)

# if time.localtime(time.time())[3]==1:
forecast_data_1 = prophet_1(data_for_model_1,holiday,city_list)
forecast_data_2 = prophet_2(data_for_model_2,holiday,city_list)
forecast_data_3 = prophet_3(data_for_model_3,city_list)

forecast_data_4 = prophet_4(data_for_model_4, city_list)

# prediction = neutralization(forecast_data_1,forecast_data_2,forecast_data_3,forecast_data_4,date=time.localtime(time.time()),citylist=city_list)
prediction,prediction_ign = neutralization_long(forecast_data_1, forecast_data_2, forecast_data_3, forecast_data_4, citylist=city_list)

    # print(prediction)