#this module runs the multi-variable linear regression engine for SO2
import sys, os
sys.path.append(os.path.abspath('..\\utility_modules'))

from metric_wrapper import metric_wrapper
from multi_lin_reg_engine import multi_lin_reg_engine

filename = "..\data\merged_new.csv"
col_list = ['SO2', 'Air Temperature', 'D-1 SO2', 'Dew Point Temperature',
            'Relative Humidity', 'Wind Speed'];
target_y = ['SO2']
test_size = 0.2

engine = multi_lin_reg_engine(filename, col_list, target_y, test_size)

multi_lin_regressor = engine.run_engine()

mae = engine.get_mean_absolute_error()
rms = engine.get_root_mean_squared_error()
accuracy = engine.get_prediction_accuracy()
score = engine.get_score()

best_model = metric_wrapper(rms, mae, accuracy, multi_lin_regressor)
print('\nbest model:')
best_model.print_values()
print('score: ', score)
