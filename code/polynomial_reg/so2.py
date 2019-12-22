# -*- coding: utf-8 -*-
import polyEngine

filename = "Data\mergedNew.csv"
col_list = ['D-1 SO2', 'Day No.']
target_y = ['SO2']
test_size = 0.2
degree = 2

n_times_run = 50
total_rms = 0
total_abs = 0
total_percentage = 0
max_percentage = 0
min_rms = 100
min_abs = 100

for i in range(n_times_run):
    engine = polyEngine.polyEngine(filename, col_list, target_y, test_size, degree)
    poly_regressor = engine.run_engine()
    
    percentage = engine.get_prediction_accuracy()
    total_rms += engine.get_root_mean_squared_error()
    total_abs += engine.get_mean_absolute_error()
    total_percentage += percentage
    
    if(min_rms > engine.get_root_mean_squared_error()):
        min_rms = engine.get_root_mean_squared_error()
    if(max_percentage < percentage):
        max_percentage = percentage
    if(min_abs > engine.get_mean_absolute_error()):
        min_abs = engine.get_mean_absolute_error()

print("avg rms: ", total_rms / n_times_run)
print("avg accuracy: ", total_percentage / n_times_run)
print("avg abs: ", total_abs / n_times_run)
print("min rms: ", min_rms)
print("max accuracy: ", max_percentage)
print("min mae: ", min_abs)