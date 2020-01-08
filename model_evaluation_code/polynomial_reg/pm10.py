#!/usr/bin/env python
# coding: utf-8
import sys, os
sys.path.append(os.path.abspath('..\\utility_modules'))

from metric_wrapper import metric_wrapper
import poly_engine

filename = "..\data\merged_new.csv"
col_list = ['PM10', 'Day No.', 'Air Temperature', 'Horizontal Visibility',
             'Dew Point Temperature', 'Year'];
target_y = ['PM10']
test_size = 0.2
degree = 2

#metric_wrapper class encapsulates error values and the regressor for a
#particular model

engine = poly_engine.poly_engine(filename, col_list, target_y, test_size, degree)

poly_regressor = engine.run_engine()

mae = engine.get_mean_absolute_error()
rms = engine.get_root_mean_squared_error()
accuracy = engine.get_prediction_accuracy()
score = engine.get_score()
best_model = metric_wrapper(rms, mae, accuracy, poly_regressor)

print('\nbest model:')
best_model.print_values()
print('score: ', score)
