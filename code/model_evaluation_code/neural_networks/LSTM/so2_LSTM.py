# This module trains a LSTM model for the prediction of SO2
# and prints the error metrics

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics

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

    # accuracy calculation(number of predictions having less than 20% error)
    for i in range(len(y_error)):
        if(y_error[i] < 0.20 * y_test[i]):
            count += 1
    print('accuracy: ', count / len(pred))

data = pd.read_csv('../../data/merged_new.csv')

X = data[['SO2', 'D-1 SO2','Air Temperature',
          'Pressure Station Level', 'Wind Speed']][:-1]
y = data[['SO2']][1:]

#split the data into training and testing sets
X_train = np.array(X[:1900])
X_test = np.array(X[1900:])
y_train = np.array(y[:1900])
y_test = np.array(y[1900:])

#scale the parameters
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#reshape the input parameters into a 3D matrix
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

#define the model structure
model = keras.models.Sequential()
model.add(keras.layers.LSTM(30, input_shape = (1, X_train.shape[1:][1]), activation = 'relu', return_sequences = True))
model.add(keras.layers.LSTM(30, activation = 'relu', return_sequences = True))
model.add(keras.layers.Dense(1))

#define the hyperparameters for fitting the data to the model
model.compile(loss = 'mean_squared_error', optimizer = keras.optimizers.Adam(0.001))

#callback function to stop running epochs when the error does not improve by a specified amount
es_callback = keras.callbacks.EarlyStopping(patience = 10, restore_best_weights = False)

#fit the model
history = model.fit(X_train, y_train, validation_split = 0.1, epochs = 10, verbose = 0, batch_size = 28,
                    callbacks = [es_callback])

#make predictions
predictions = model.predict(X_test)
predictions = predictions.flatten()
print_metrics(predictions, y_test)

#print training error for checking over-fitting
mse_train = metrics.mean_squared_error(y_train, (model.predict(X_train)).flatten())
print('train mse: ', mse_train)

