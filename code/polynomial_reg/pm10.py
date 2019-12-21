#!/usr/bin/env python
# coding: utf-8

import polyEngine


filename = "Data\merged_data_DminusOne.csv"
col_list = ['D-1 PM10', 'Day No.', 'Air Temperature', 'Horizontal Visibility',
            'Pressure Station Level', 'Relative Humidity', 'Dew Point Temperature'];
target_y = ['PM10']
test_size = 0.2
degree = 2
total_rms = 0
n_times_run = 10
total_percentage = 0

for i in range(n_times_run):
    engine = polyEngine.polyEngine(filename, col_list, target_y, test_size, degree)
    poly_regressor = engine.runEngine()
    error_list, rms, percentage = engine.getStatusLog()
    total_rms += rms
    total_percentage += percentage

print(total_rms / n_times_run)
print(total_percentage / n_times_run)
