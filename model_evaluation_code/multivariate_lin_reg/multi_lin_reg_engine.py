# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LinearRegression
#from sklearn.model_selection import train_test_split

class multi_lin_reg_engine:
    def __init__(self, filename, col_list, target_y, test_size):
        self.filename = filename
        self.col_list = col_list
        self.target_y = target_y
        self.test_size = test_size
        self.dataset = self.load_data()
        self.actual_values = []
        self.predicted_values = []
        self.score = 0.0

    def load_data(self):
        data = pd.read_csv(self.filename)
        return data.iloc[:2372, : ]

    def run_engine(self):
        y = self.dataset[self.target_y][1:].values
        X = self.dataset[self.col_list][:-1].values

        #split the entire set into training and test sets
        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = self.test_size)
        
        train_length = round(X.shape[0] * (1 - self.test_size))
        
        X_train = np.array(X[:train_length])
        y_train = np.array(y[:train_length])
        X_test = np.array(X[train_length:])
        y_test = np.array(y[train_length:])
        
        self.actual_values = y_test
        
        lin_reg = LinearRegression()
        #fit the prediction model
        lin_reg.fit(X_train, y_train)
        
        self.score = lin_reg.score(X_test, self.actual_values)
        self.predict(lin_reg, X_test)
        return lin_reg

    def predict(self, lin_reg, X):
        predictions = lin_reg.predict(X);
        self.predicted_values = predictions
    
    def get_score(self):
        return self.score
    
    def get_root_mean_squared_error(self):
        rmse = np.sqrt(metrics.mean_squared_error(self.actual_values, self.predicted_values))
        return rmse
    
    def get_mean_absolute_error(self):
        mae = metrics.mean_absolute_error(self.actual_values, self.predicted_values)
        return mae
    
    def get_root_mean_squared_log_error(self):
        rmsle = np.sqrt(metrics.mean_squared_log_error(self.actual_values, self.predicted_values))
        return rmsle
    
    def get_error(self):
        return self.predicted_values - self.actual_values

    def get_prediction_accuracy(self):
        #calculate errors for each prediction
        y_error = self.predicted_values - self.actual_values
        #variable count stores errors in prediction where error < 20%
        count = []
        for i in range(len(self.actual_values)):
            if(abs(y_error[i]) < 0.20 * self.actual_values[i]):
                count.append(abs(y_error))
        return len(count) / len(y_error) * 100

