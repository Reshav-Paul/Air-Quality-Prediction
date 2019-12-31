import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

class forest:
    def __init__(self, filename, col_list, target_y, test_size, no_of_trees):
        self.filename = filename
        self.col_list = col_list
        self.target_y = target_y
        self.test_size = test_size
        self.dataset = self.load_data(filename)
        self.no_of_trees = no_of_trees
        self.actual_values = []
        self.predicted_values = []
        self.test_values = []

    def load_data(self, fileName):
        data = pd.read_csv(self.filename)
        return data.iloc[:2372, : ]

    def run_engine(self):
        X = self.dataset[self.col_list][:-1]
        y = self.dataset[self.target_y][1:]
        
        #Split the entire set into training and test set
        
        #x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = self.test_size)
        
        train_length = round(X.shape[0] * (1 - self.test_size))
        
        x_train = X[:train_length]
        y_train = y[:train_length]
        x_test = X[train_length:]
        y_test = y[train_length:]
        
        
        
        xx_train = np.array(x_train)
        yy_train = np.array(y_train)
        xx_test = np.array(x_test)
        yy_test = np.array(y_test)
        self.actual_values = yy_test
        self.test_values = xx_test
        yy_train = np.ravel(yy_train)
        #print(self.actual_values.shape)
        
        #Fitting the Random Forest regression to the data set
        
        regressor = RandomForestRegressor(n_estimators = self.no_of_trees)
        regressor.fit(xx_train, yy_train)
        
        self.predict(regressor)
        return regressor

    def predict(self, regressor):
        self.predicted_values = regressor.predict(self.test_values);
        self.predicted_values = self.predicted_values.reshape(self.predicted_values.shape[0], 1)
        
    def get_score(self, regressor):        
        return(regressor.score(self.test_values, self.actual_values))
        
    def get_mean_absolute_error(self):
        mae = metrics.mean_absolute_error(self.actual_values, self.predicted_values)
        return mae
    
    def get_mean_squared_error(self):
        mse = metrics.mean_squared_error(self.actual_values, self.predicted_values)
        return mse

    def get_root_mean_squared_error(self):
        rmse = np.sqrt(metrics.mean_squared_error(self.actual_values, self.predicted_values))
        return rmse

    def get_root_mean_squared_log_error(self):
        rmsle = np.sqrt(metrics.mean_squared_log_error(abs(self.actual_values), abs(self.predicted_values)))
        return rmsle

    def get_prediction_accuracy(self):
        y_error = self.predicted_values - self.actual_values
        count = []
        for i in range(len(self.actual_values)):
            if((abs(y_error[i])) < (0.2 * self.actual_values[i])):
                count.append(abs(y_error[i]))
        return (len(count) / len(y_error) * 100)

    def get_error(self):
        y_error = self.predicted_values - self.actual_values
        return y_error

    
   
