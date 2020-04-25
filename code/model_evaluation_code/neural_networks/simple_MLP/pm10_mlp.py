# This module trains a Multi-layer perceptron model for PM10
# and prints the error metrics

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn import metrics
import matplotlib.pyplot as plt

def get_performance(model, x_val, y_val):
    #mean absolute error calculation
    mae = metrics.mean_absolute_error(y_val, model.predict(x_val))

    #mean squared error calculation
    mse = metrics.mean_squared_error(y_val, model.predict(x_val))

    #score calculation
    rsq = metrics.r2_score(y_val, model.predict(x_val))
    return (mae, mse, rsq)

def get_accuracy(model, xx_test, yy_test):
    #accuracy calculation(number of predictions having less than 20% error)
    actual_val = yy_test
    predicted_val = model.predict(xx_test)
    predicted_val = predicted_val.reshape(predicted_val.shape[0], 1)
    error_val = predicted_val - actual_val
    count = []
    for i in range(len(actual_val)):
      if((abs(error_val[i])) < (0.2 * actual_val[i])):
        count.append(abs(error_val[i]))
    return(len(count) / len(error_val) * 100)

def get_plot(model, x_val, y_val):
    #visualizing performance
    y_pred = model.predict(x_val)
    x = np.arange(len(y_val))
    plt.figure(figsize = (10, 10))
    plt.scatter(x, y_val, label = "actual", c = "b", s = 100)
    plt.scatter(x, y_pred, label = "prediction", c = "orange", s = 100)
    plt.xlabel("Days", fontsize = 13)
    plt.ylabel("PM10", fontsize = 13)
    plt.xticks(size = 10)
    plt.yticks(size = 10)
    plt.title("Neural Network : PM10 scatter plot", fontsize = 15)
    plt.legend(fontsize = 12)
    plt.show()

#importing dataset

data_set = pd.read_csv('../../data/merged_new.csv')
data_set.head(3)

#selecting independent and dependent/target attributes

indep_list = ['Air Temperature', 'Relative Humidity', 'Horizontal Visibility', 'Year', 'D-1 PM10', 'Day No.', 'PM10']
dep_list = ['PM10']

#setting up independent and dependent/target variables

indep_data_set = data_set[indep_list][:-1]
dep_data_set = data_set[dep_list][1:]

#splitting the dataset into training and test set

calc_test_size = round(indep_data_set.shape[0]*(1 - 0.2))
x_train = indep_data_set.iloc[:calc_test_size, :]
x_test = indep_data_set.iloc[calc_test_size:, :]
y_train = dep_data_set.iloc[:calc_test_size, :]
y_test = dep_data_set.iloc[calc_test_size:, :]
xx_train = np.array(x_train)
yy_train = np.array(y_train)
xx_test = np.array(x_test)
yy_test = np.array(y_test)

#feature scaling on the training and test set

sc = MinMaxScaler()
sc.fit(xx_train)
xx_train = sc.transform(xx_train)
xx_test = sc.transform(xx_test)

#visualizing datasets after feature scaling

pd.DataFrame(xx_train).describe()
pd.DataFrame(xx_test).describe()

#initializing the model and setting up the parameters

'''model structure :(Three hidden layers having 75 neurons for each,
activation function for all hidden layers : rectifier(relu),
epoch value: 200, optimizer function : lbfgs, learning rate : 0.003 and batch size : 28)'''

nn = MLPRegressor(hidden_layer_sizes=(75, 75, 75, ), activation='relu', max_iter= 200, solver='lbfgs', n_iter_no_change=100, learning_rate_init=0.003, batch_size=28)

#fitting training dataset to the model

nn.fit(xx_train, yy_train)

#evaluating performance of the model on training set

mae, mse, rsq = get_performance(nn, xx_train, yy_train)

#showing performance on training set

print(mae, mse, rsq)

#evaluating performance of the model on test set

maet, mset, rsqt = get_performance(nn, xx_test, yy_test)

#showing performance on test set

print(maet, mset, rsqt)

#visualizing performance on traing set

get_plot(nn, xx_train, yy_train)

#visualizing performance on test set

get_plot(nn, xx_test, yy_test)

#prediction

predicted_val = nn.predict(xx_test)

#accuracy of prediction

acc = get_accuracy(nn, xx_test, yy_test)


