# This module runs the SVR engine for SO2

import sys, os
sys.path.append(os.path.abspath('..\\utility_modules'))

from svr_engine import svr_engine
from metric_wrapper import metric_wrapper

filename = "..\data\merged_new.csv"
col_list = ['SO2', 'Day No.', 'D-1 SO2', 'Pressure Station Level', 'Relative Humidity'];
target_y = ['SO2']
test_size = 0.2
kernel = 'rbf'

engine = svr_engine(filename, col_list, target_y, kernel, test_size)

regressor = engine.run_engine()
    
mae = engine.get_mean_absolute_error()
rms = engine.get_root_mean_squared_error()
accuracy = engine.get_prediction_accuracy()
score = engine.get_score()

best_model = metric_wrapper(rms, mae, accuracy, regressor)

print('\nbest model:')
best_model.print_values()
print('score: ', score)