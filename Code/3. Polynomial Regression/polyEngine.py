#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# In[3]:


class polyEngine:
    def __init__(self, filename, col_list, target_y, test_size, degree):
        self.filename = filename
        self.col_list = col_list
        self.target_y = target_y
        self.test_size = test_size
        self.degree = degree
        self.dataset = self.loadData(filename)
        self.actual_values = []
        self.predicted_values = []

    def loadData(self, fileName):
        data = pd.read_csv(self.filename)
        return data.iloc[:2372, : ]

    def runEngine(self):
        y = self.dataset[self.target_y].values
        y = np.array(y)
        X = self.dataset[self.col_list].values
        X = np.array(X)

        #split the entire set into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = self.test_size)
        self.actual_values = y_test
        #fit the X parameters of the training set into a polynomial of degree d, best case d = 2
        poly_reg = PolynomialFeatures(degree = self.degree)
        X_poly = poly_reg.fit_transform(X_train)
        lin_reg2 = LinearRegression()

        #fit the prediction model
        lin_reg2.fit(X_poly, y_train)
        
        X_transform = poly_reg.fit_transform(X_test)
        self.predict(lin_reg2, X_transform)
        return lin_reg2

    def predict(self, lin_reg2, X):
        predictions = lin_reg2.predict(X);
        self.predicted_values = predictions

    def getStatusLog(self):
        #calculate errors for each prediction
        y_error = self.predicted_values - self.actual_values
        
        #mean sqaure error
        print("rms error: ", end = ": ")
        print(np.sqrt(metrics.mean_squared_error(self.actual_values, self.predicted_values)))
        
        #variable count stores errors in prediction where error < 20%
        count = []
        for i in range(len(self.actual_values)):
            if(abs(y_error[i]) < 0.20 * self.actual_values[i]):
                count.append(abs(y_error))
        
        #print the percentage of predictions where error < 20%        
        print("percentage of predictions with error < 20%", end = ": ")
        print(len(count) / len(y_error) * 100)
        
        return y_error

