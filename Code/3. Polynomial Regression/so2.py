# -*- coding: utf-8 -*-
import polyEngine

filename = "Data\merged_data_DminusOne.csv"
col_list = ['Air Temperature','Pressure Station Level', 'D-1 SO2']
target_y = ['SO2']
test_size = 0.2
degree = 2
n_times_run = 10
total_percentage = 0
total_rms = 0

for i in range(n_times_run):
    engine = polyEngine.polyEngine(filename, col_list, target_y, test_size, degree)
    poly_regressor = engine.runEngine()
    error_list, rms, percentage = engine.getStatusLog()
    total_rms += rms
    total_percentage += percentage

print(total_rms / n_times_run)
print(total_percentage / n_times_run)