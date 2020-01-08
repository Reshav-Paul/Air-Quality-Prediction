import sys, os
sys.path.append(os.path.abspath('..\\utility_modules'))

from metric_wrapper import metric_wrapper
from tree_engine import tree_engine

col_list = ['SO2']
target = ['SO2']
test_size = 0.2
file_name = '..\\data\\merged_new.csv'

engine = tree_engine(file_name, col_list, target, test_size)

tree_regressor = engine.run_engine()
    
score = engine.get_score(tree_regressor)
mae = engine.get_mean_absolute_error()
rmse = engine.get_root_mean_squared_error()
accuracy = engine.get_prediction_accuracy()
best_model = metric_wrapper(rmse, mae, accuracy, tree_regressor)

print('\nbest model:')
best_model.print_values()
print('score: ', score)