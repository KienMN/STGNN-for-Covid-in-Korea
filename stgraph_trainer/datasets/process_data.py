import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def data_diff(data):
  """
  Calculates the difference of a Dataframe element compared with previous element in the Dataframe.

  Parameters
  ----------
  data: 2D array
    Input array data.

  Returns
  -------
  data: DataFrame
    Difference array data, or the changes to the observations from one to the next.
  """
  if not isinstance(data, pd.DataFrame):
    data = pd.DataFrame(data)
  data = data.diff()
  data = data.iloc[1:, :]
  return data

def inverse_diff(value, raw_data, interval=0):
  """
  Inverse difference, given difference value and raw data of previous row.

  Parameters
  ----------
  value: float or array
    The difference value.

  raw_data: array
    The original array of data.

  interval: int
    The number of rows from the last row of raw_data.

  Returns
  -------
  original_value: float or array
    The original value (before differencing) of (-interval + 1) row.
  """
  return value + raw_data[-interval]

def timeseries_to_supervised(data, lag=1):
  """
  Turn time series data to supervised data for machine learning model.

  Parameters
  ----------
  data: 2D array shape of (n_samples, n_timeseries)
    The original time serires data. n_timeseires is the number of time series in the data.

  lags: int, defautl: 1
    The number of time steps (or context length) of the series to be processed to make prediction.

  Returns
  -------
  output: 3D array shape of (n_samples - lags, lags + 1, n_timeseries)
    The supervised sequence data for machine learning model.
  """
  if not isinstance(data, pd.DataFrame):
    data = pd.DataFrame(data)
  dfs = [data.shift(i).iloc[lag:].values for i in range(lag, -1, -1)]
  return np.array(dfs).transpose(1, 0, 2)
  
def preprocess_data_for_lstm_model(data,
                                   split_date,
                                   time_steps,
                                   feature_range=None):
  """
  Prepare data for LSTM model.

  Parameters
  ----------
  data: 2D array shape of (n_samples, n_timeseries)
    The original time serires data. n_timeseires is the number of time series in the data.

  split_date: str, format of YYYY-MM-DD
    The date to split data into train and test set.

  time_steps: int
    The number of time steps (or context length) of the series to be processed to make prediction.

  feature_range: tuple, defautl: None
    The min, max value for using MinMaxScaler to normalize data.
    Default is None, which means using StandardScaler.

  Returns
  -------
  X_train: 3D array shape of (n_train_samples - time_steps, time_steps, n_timeseries)
    The supervised train data.

  y_train: 2D array shape of (n_train_samples - time_steps, n_timeseries)
    The supervised train labels.

  X_test: 3D array shape of (n_test_samples, time_steps, n_timeseries)
    The supervised test data.

  y_test: 2D array shape of (n_test_samples, n_timeseries)
    The supervised test labels.

  train: 2D array shape of (n_train_samples, n_timeseries)
    The value of train dataset, after differencing.

  test: 2D array shape of (n_test_samples + time_steps, n_timeseries)
    The value of test dataset, after differencing.

  scaler: object
    MinMaxScaler or Standard Scaler used to transform the data.
  """
  
  data = data_diff(data)

  train = data[data.index < split_date]
  test = data.iloc[len(train) - time_steps:, :]

  if feature_range is not None:
    scaler = MinMaxScaler(feature_range=feature_range)
  else:
    scaler = StandardScaler()
  train_scaled = scaler.fit_transform(train)
  test_scaled = scaler.transform(test)

  train_arr = timeseries_to_supervised(train_scaled, lag=time_steps)
  X_train = train_arr[:, :-1, :]
  y_train = train_arr[:, -1, :]

  test_arr = timeseries_to_supervised(test_scaled, lag=time_steps)
  X_test = test_arr[:, :-1, :]
  y_test = test_arr[:, -1, :]

  return X_train, y_train, X_test, y_test, train, test, scaler

def preprocess_data_for_seq2seq(data,
                                split_date,
                                time_steps,
                                feature_range=None):
  """
  Prepare data for Seq2Seq model.

  Parameters
  ----------
  data: 2D array shape of (n_samples, n_timeseries)
    The original time serires data. n_timeseires is the number of time series in the data.

  split_date: str, format of YYYY-MM-DD
    The date to split data into train and test set.

  time_steps: int
    The number of time steps (or context length) of the series to be processed to make prediction.

  feature_range: tuple, defautl: None
    The min, max value for using MinMaxScaler to normalize data.
    Default is None, which means using StandardScaler.
    The same scaler is applied for all data, not for each column.

  Returns
  -------
  X_train: 2D array shape of ((n_train_samples - time_steps) * n_timeseries, time_steps)
    The supervised train data.

  y_train: 2D array shape of ((n_train_samples - time_steps) * n_timeseries, 1)
    The supervised train labels.

  X_test: 2D array shape of (n_test_samples * n_timeseries, time_steps)
    The supervised test data.

  y_test: 2D array shape of (n_test_samples * n_timeseries, 1)
    The supervised test labels.

  train: 2D array shape of (n_train_samples, n_timeseries)
    The value of train dataset, after differencing.

  test: 2D array shape of (n_test_samples + time_steps, n_timeseries)
    The value of test dataset, after differencing.

  scaler: object
    MinMaxScaler or Standard Scaler used to transform the data.
  """
  data = data_diff(data)

  train = data[data.index < split_date]
  test = data.iloc[len(train) - time_steps:, :]

  if feature_range is not None:
    scaler = MinMaxScaler(feature_range=feature_range)
  else:
    scaler = StandardScaler()
  
  train_scaled = scaler.fit_transform(train.values.reshape([-1, 1]))
  train_scaled = train_scaled.reshape(train.shape)
  test_scaled = scaler.transform(test.values.reshape([-1, 1]))
  test_scaled = test_scaled.reshape(test.shape)

  train_arr = timeseries_to_supervised(train_scaled, lag=time_steps)
  X_train = train_arr[:, :-1, :]
  y_train = train_arr[:, -1, :]

  test_arr = timeseries_to_supervised(test_scaled, lag=time_steps)
  X_test = test_arr[:, :-1, :]
  y_test = test_arr[:, -1, :]

  X_train = np.concatenate([X_train[:, :, i] for i in range(X_train.shape[-1])])
  y_train = np.concatenate([y_train[:, [i]] for i in range(y_train.shape[-1])])

  X_test = np.concatenate([X_test[:, :, i] for i in range(X_test.shape[-1])])
  y_test = np.concatenate([y_test[:, [i]] for i in range(y_test.shape[-1])])

  return X_train, y_train, X_test, y_test, train, test, scaler

def preprocess_data_for_stgnn(data,
                              split_date,
                              time_steps,
                              feature_range=None):
  """
  Prepare data for STGNN model.

  Parameters
  ----------
  data: 2D array shape of (n_samples, n_timeseries)
    The original time serires data. n_timeseires is the number of time series in the data.

  split_date: str, format of YYYY-MM-DD
    The date to split data into train and test set.

  time_steps: int
    The number of time steps (or context length) of the series to be processed to make prediction.

  feature_range: tuple, defautl: None
    The min, max value for using MinMaxScaler to normalize data.
    Default is None, which means using StandardScaler.

  Returns
  -------
  X_train: 3D array shape of (n_train_samples - time_steps, n_timeseries, time_steps)
    The supervised train data.

  y_train: 2D array shape of (n_train_samples - time_steps, n_timeseries)
    The supervised train labels.

  X_test: 3D array shape of (n_test_samples, n_timeseries, time_steps)
    The supervised test data.

  y_test: 2D array shape of (n_test_samples, n_timeseries)
    The supervised test labels.

  train: 2D array shape of (n_train_samples, n_timeseries)
    The value of train dataset, after differencing.

  test: 2D array shape of (n_test_samples + time_steps, n_timeseries)
    The value of test dataset, after differencing.

  scaler: object
    MinMaxScaler or Standard Scaler used to transform the data.
  """
  
  data = data_diff(data)

  train = data[data.index < split_date]
  test = data.iloc[len(train) - time_steps:, :]

  if feature_range is not None:
    scaler = MinMaxScaler(feature_range=feature_range)
  else:
    scaler = StandardScaler()
  train_scaled = scaler.fit_transform(train)
  test_scaled = scaler.transform(test)

  train_arr = timeseries_to_supervised(train_scaled, lag=time_steps)
  X_train = train_arr[:, :-1, :].transpose(0, 2, 1)
  y_train = train_arr[:, -1, :]

  test_arr = timeseries_to_supervised(test_scaled, lag=time_steps)
  X_test = test_arr[:, :-1, :].transpose(0, 2, 1)
  y_test = test_arr[:, -1, :]

  return X_train, y_train, X_test, y_test, train, test, scaler