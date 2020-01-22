import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics

from tensorflow import keras

def print_metrics(pred, y_vals):
    print('mae: ', metrics.mean_absolute_error(y_vals, pred))
    print('mse: ', metrics.mean_squared_error(y_vals, pred))
    print('rmse: ', np.sqrt(metrics.mean_squared_error(y_vals, pred)))
    print('score: ', metrics.r2_score(y_vals, pred))
    count = 0
    y_error = pred.flatten() - y_vals.flatten()
    y_error = np.array([abs(e) for e in y_error]).flatten()
    for i in range(len(y_error)):
        if(y_error[i] < 0.20 * y_test[i]):
            count += 1
    print('accuracy: ', count / len(pred))

data = pd.read_csv('../../data/merged_new.csv')

X = data[['SO2', 'D-1 SO2', 'Dew Point Temperature', 'Relative Humidity']][:-1]
y = data[['SO2']][1:]

X_train = np.array(X[:1900])
X_test = np.array(X[1900:])
y_train = np.array(y[:1900])
y_test = np.array(y[1900:])

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = keras.models.Sequential()
model.add(keras.layers.Dense(10, input_shape = X_train.shape[1:], activation = 'relu'))
model.add(keras.layers.Dense(10, activation = 'relu'))
model.add(keras.layers.Dense(1))

model.compile(loss = 'mean_squared_error', optimizer = keras.optimizers.Adam(0.001))

history = model.fit(X_train, y_train, validation_split = 0.1,
                    epochs = 50, verbose = 0)

predictions = model.predict(X_test)
print_metrics(predictions, y_test)

train_pred = model.predict(X_train)
print('train mse: ', metrics.mean_squared_error(y_train, train_pred))

