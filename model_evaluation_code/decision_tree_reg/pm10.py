import sys, os
sys.path.append(os.path.abspath('..\\utility_modules'))

from metric_wrapper import metric_wrapper
from tree_engine import tree_engine

col_list = ['PM10', 'Day No.', 'Air Temperature', 'Horizontal Visibility',
           'Relative Humidity']
target = ['PM10']
test_size = 0.2
file_name = '..\\data\\merged_new.csv'

n_times_run = 100
total_rmse = 0
total_accuracy = 0
total_score = 0
total_mae = 0

least_mae_model = metric_wrapper(100, 100, 0)
least_rmse_model = metric_wrapper(100, 100, 0)
greatest_accuracy_model = metric_wrapper(100, 100, 0)

engine = tree_engine(file_name, col_list, target, test_size)

for i in range(n_times_run):
    
    tree_regressor = engine.run_engine()
    
    score = engine.get_score(tree_regressor)
    mae = engine.get_mean_absolute_error()
    rmse = engine.get_root_mean_squared_error()
    accuracy = engine.get_prediction_accuracy()
    
    total_rmse += rmse
    total_score += score
    total_accuracy += accuracy
    total_mae += mae
    
    if(mae < least_mae_model.mae):
        least_mae_model.copy(rmse, mae, accuracy, tree_regressor)
    if(rmse < least_rmse_model.rmse):
        least_rmse_model.copy(rmse, mae, accuracy, tree_regressor)
    if(accuracy > greatest_accuracy_model.accuracy):
        greatest_accuracy_model.copy(rmse, mae, accuracy, tree_regressor)
    
print("average mae : ", total_mae / n_times_run)
print("average rmse : ", total_rmse / n_times_run)
print("average accuracy : ", total_accuracy / n_times_run)
print("average score : ", total_score / n_times_run)

print('\nleast mae model:')
least_mae_model.print_values()

print('\nleast rmse model:')
least_rmse_model.print_values()

print('\ngreatest accuracy model:')
greatest_accuracy_model.print_values()