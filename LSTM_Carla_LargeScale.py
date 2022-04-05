"""
Aims for:
1. Training LSTM with multiple different datasets (having different start/end time but with the same time step)
2. Predicting one whole dataset

Usage:
1.Locate to a directory file on line 79
2.Put all wanted training datasets in front of the order of files
3.Put the one testing dataset at last
e.g.
'LSTM File'
    -Train1.csv
    -Train2.csv
    -Train3.csv
    -Train4.csv
    -Test.csv
"""
import os
import glob
from matplotlib import pyplot
from numpy import concatenate
from pandas import DataFrame
from pandas import concat
import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

"""
------------------------------------------------------------------------------------------------
url: https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/
Frame a time series as a supervised learning dataset.
Arguments:
"data":     Sequence of observations as a list or NumPy array.
"n_in":     Number of lag observations as input (X).
"n_out":    Number of observations as output (y).
"dropnan":  Boolean whether or not to drop rows with NaN values.
Returns:
"Pandas DataFrame of series framed for supervised learning."
------------------------------------------------------------------------------------------------
"""


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# fix random seed for reproducibility
numpy.random.seed(7)

# load the dataset
df_raw = DataFrame()
path = os.getcwd()
csv_files = glob.glob(os.path.join(path, 'Test/*.csv'))
print(csv_files[-1])
for file in csv_files:
    df_raw = DataFrame()
    raw = read_csv(file)
    # print('File Names:', file.split("\\")[-1])
    df_raw['x_coor'] = raw.x_coor
    df_raw['y_coor'] = raw.y_coor
    df_raw['Yaw'] = raw.Yaw
    df_raw['Speed'] = raw.Speed
    # df_raw['Time'] = raw.Time
    values = df_raw.values
    dataset = series_to_supervised(values)
    dataset = dataset.astype('float32')
    # drop columns we don't want to predict
    dataset.drop(dataset.columns[[6, 7]], axis=1, inplace=True)
    # print(dataset)

    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)
    # print(dataset.shape)  # Shape = (xxxx, 6)

    # split into train and test sets
    # if file != csv_files[-1]:
    train_size = int(len(dataset) * 0.67)
    train = dataset[:train_size, :]
    test = dataset[train_size:, :]

    # split into input and outputs
    train_X, train_y = train[:, :-2], train[:, -2:]
    test_X, test_y = test[:, :-2], test[:, -2:]

    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

    # design network
    if file == csv_files[0]:
        model = Sequential()
        print(train_X.shape[1], train_X.shape[2])
        model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
        model.add(Dense(2))
        model.compile(loss='mae', optimizer='adam')

    # train network
    if file != csv_files[-1]:
        history = model.fit(train_X, train_y, epochs=50, batch_size=64, validation_data=(test_X, test_y)
                            , verbose=2, shuffle=False)
    else:
        # make a prediction
        predict_X = numpy.concatenate((train_X, test_X))
        # predict_X = test_X
        predict_Y = numpy.concatenate((train_y, test_y))
        # predict_Y = test_y
        # print(predict_X)
        yhat = model.predict(predict_X)  # shape of (351, 2)
        predict_X = predict_X.reshape((predict_X.shape[0], predict_X.shape[2]))

        # invert scaling for forecast
        inv_yhat = concatenate((predict_X, yhat), axis=1)
        # [x(t-1) y(t-1) Yaw(t) Speed(t) predicted_x(t) predicted_y(t)]
        yhat = scaler.inverse_transform(inv_yhat)
        inv_yhat = scaler.inverse_transform(inv_yhat)
        # print(inv_yhat)
        inv_yhat_x_coor = inv_yhat[:, 4]
        inv_yhat_y_coor = inv_yhat[:, 5]

        # invert scaling for actual
        inv_y = concatenate((predict_X, predict_Y), axis=1)
        inv_y = scaler.inverse_transform(inv_y)
        # print(inv_y)
        inv_y_x_coor = inv_y[:, 4]
        inv_y_y_coor = inv_y[:, 5]

        # calculate RMSE
        rmse_x = math.sqrt(mean_squared_error(inv_y_x_coor, inv_yhat_x_coor))
        rmse_y = math.sqrt(mean_squared_error(inv_y_y_coor, inv_yhat_y_coor))
        print('Test RMSE of x_coordinate: %.3f' % rmse_x)
        print('Test RMSE of y_coordinate: %.3f' % rmse_y)

        # plot predicted and the actual coordinates

        figure, axis = plt.subplots(2)
        figure.suptitle("LSTM Coordinates Prediction\n(Test: 500ms Train: 0ms)", fontsize=13, fontweight="bold")

        axis[0].plot(inv_y_x_coor, c='black')
        axis[0].plot(inv_yhat_x_coor, '--', c='g')
        axis[0].legend(["Actual", "Predicted"])
        axis[0].set_title("X coordinates (RMSE=%.3f)" % rmse_x, fontsize=10)

        axis[1].plot(inv_y_y_coor, c='black')
        axis[1].plot(inv_yhat_y_coor, '--', c='g')
        axis[1].legend(["Actual", "Predicted"])
        axis[1].set_title("Y coordinates (RMSE=%.3f)" % rmse_y, fontsize=10)
        plt.show()


