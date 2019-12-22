# -*- coding: utf-8 -*-

from svr_engine import svr_engine

filename = "..\data\merged_new.csv"
col_list = ['D-1 PM10', 'Day No.', 'Air Temperature', 'Horizontal Visibility',
             'Dew Point Temperature'];
target_y = ['PM10']
test_size = 0.2
kernel = 'rbf'
engine = svr_engine(filename, col_list, target_y, kernel, test_size)
regressor = engine.run_engine()
print(engine.get_mean_absolute_error())
print(engine.get_root_mean_squared_error())
print(engine.get_prediction_accuracy())