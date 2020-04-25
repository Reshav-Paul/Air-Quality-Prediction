# This module trains a Wide-Deep Multi-layer perceptron model for SO2
# prints the error metrics

import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler

from tensorflow import keras

#function for printing the metrics of the predictions
def print_metrics(pred, y_vals):
    print('mae: ', metrics.mean_absolute_error(y_vals, pred))
    print('mse: ', metrics.mean_squared_error(y_vals, pred))
    print('rmse: ', np.sqrt(metrics.mean_squared_error(y_vals, pred)))
    print('score: ', metrics.r2_score(y_vals, pred))
    count = 0
    y_error = pred.flatten() - y_vals.flatten()
    y_error = np.array([abs(e) for e in y_error]).flatten()

    #accuracy calculation(number of predictions having less than 20% error)
    for i in range(len(y_error)):
        if(y_error[i] < 0.20 * y_test[i]):
            count += 1
    print('accuracy: ', count / len(pred))

data = pd.read_csv('../../data/merged_new.csv')

# split the data into training and testing sets
# a sets are the deep parameters and b sets are the wide parameters
X_a = data[['SO2', 'D-1 SO2']][:-1]
X_b = data[['SO2']][:-1]
y = data[['SO2']][1:]

X_train_a = np.array(X_a[:1900])
X_train_b = np.array(X_b[:1900])

X_test_a = np.array(X_a[1900:])
X_test_b = np.array(X_b[1900:])

y_train = np.array(y[:1900])
y_test = np.array(y[1900:])

#scale the wide and deep parameters
a_scaler = MinMaxScaler()
b_scaler = MinMaxScaler()

X_train_a = a_scaler.fit_transform(X_train_a)
X_train_b = b_scaler.fit_transform(X_train_b)

X_test_a = a_scaler.transform(X_test_a)
X_test_b = b_scaler.transform(X_test_b)

#define the model structure

# deep input
input_layer_a = keras.layers.Input(shape = X_train_a.shape[1:])

# wide input
input_layer_b = keras.layers.Input(shape = X_train_b.shape[1:])

hidden1 = keras.layers.Dense(30, activation = 'relu')(input_layer_a)
hidden2 = keras.layers.Dense(30, activation = 'relu')(hidden1)
hidden3 = keras.layers.Dense(30, activation = 'relu')(hidden2)

#concat the wide and deep layers
concat = keras.layers.concatenate([input_layer_b, hidden3])
output_layer = keras.layers.Dense(1)(concat)

model = keras.models.Model(inputs = [input_layer_a, input_layer_b], outputs = [output_layer])

#define the hyperparameters for fitting the data to the model
model.compile(optimizer = keras.optimizers.Adam(0.001), loss = 'mse', metrics = ['mse'])

#callback function to stop fitting when the error does not change by a specific value
es_callback = keras.callbacks.EarlyStopping(patience = 20, restore_best_weights = False)

#fit the model
history = model.fit((X_train_a, X_train_b), y_train, epochs = 50,
                    validation_split = 0.1, verbose = 0, batch_size = 28, callbacks = [es_callback])

#make predictions
train_predictions = model.predict((X_train_a, X_train_b))
predictions = model.predict((X_test_a, X_test_b))
print_metrics(predictions, y_test)

#print training error to avoid overfitting
print('train mse: ', metrics.mean_squared_error(y_train, train_predictions))
