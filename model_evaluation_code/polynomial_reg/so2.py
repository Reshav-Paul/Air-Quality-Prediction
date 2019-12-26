# -*- coding: utf-8 -*-
import sys, os
sys.path.append(os.path.abspath('..\\utility_modules'))

from metric_wrapper import metric_wrapper
import poly_engine

filename = "..\data\merged_new.csv"
col_list = ['D-1 SO2', 'Day No.']
target_y = ['SO2']
test_size = 0.2
degree = 2

n_times_run = 50
total_rms = 0
total_mae = 0
total_accuracy = 0
total_score = 0

#metric_wrapper class encapsulates error values and the regressor for a
#particular model
least_mae_model = metric_wrapper(100, 100, 0)
least_rmse_model = metric_wrapper(100, 100, 0)
greatest_accuracy_model = metric_wrapper(100, 100, 0)

engine = poly_engine.poly_engine(filename, col_list, target_y, test_size, degree)

for i in range(n_times_run):
    
    poly_regressor = engine.run_engine()
    
    mae = engine.get_mean_absolute_error()
    rms = engine.get_root_mean_squared_error()
    accuracy = engine.get_prediction_accuracy()
    
    total_rms += rms
    total_mae += mae
    total_accuracy += accuracy
    total_score += engine.get_score()
    
    if(mae < least_mae_model.mae):
        least_mae_model.copy(rms, mae, accuracy, poly_regressor)
    if(rms < least_rmse_model.rmse):
        least_rmse_model.copy(rms, mae, accuracy, poly_regressor)
    if(accuracy > greatest_accuracy_model.accuracy):
        greatest_accuracy_model.copy(rms, mae, accuracy, poly_regressor)

print('avg mae: ', total_mae / n_times_run)
print('avg rmse: ', total_rms / n_times_run)
print('avg accuracy: ', total_accuracy / n_times_run)
print('avg score: ', total_score / n_times_run)

print('\nleast mae model:')
least_mae_model.print_values()

print('\nleast rmse model:')
least_rmse_model.print_values()

print('\ngreatest accuracy model:')
greatest_accuracy_model.print_values()