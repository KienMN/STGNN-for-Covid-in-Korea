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
  return value + raw_data[-interval]

def timeseries_to_supervised(data, lag=1):
  if not isinstance(data, pd.DataFrame):
    data = pd.DataFrame(data)
  dfs = [data.shift(i).iloc[lag:].values for i in range(lag, -1, -1)]
  return np.array(dfs).transpose(1, 0, 2)
  
def preprocess_data_for_lstm_model(data, split_date, time_steps, feature_range=(-1, 1)):
  data = data_diff(data)

  train = data[data.index < split_date]
  test = data.iloc[len(train) - time_steps:, :]

  # scaler = MinMaxScaler(feature_range=feature_range)
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

def preprocess_data_for_seq2seq(data, split_date, time_steps, feature_range=(-1, 1)):
  data = data_diff(data)

  train = data[data.index < split_date]
  test = data.iloc[len(train) - time_steps:, :]

  # scaler = MinMaxScaler(feature_range=feature_range)
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