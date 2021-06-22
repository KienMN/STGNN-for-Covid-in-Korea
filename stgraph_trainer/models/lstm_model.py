import tensorflow as tf

def create_lstm_model(lstm_units,
                      output_size,
                      batch_size,
                      time_steps,
                      n_features,
                      dropout=0.,
                      recurrent_dropout=0.):
  """
  Create a LSTM model.

  Parameters
  ----------
  lstm_units: int
    The number of units of LSTM's output space.

  output_size: int
    The number of units in final dense layer.

  batch_size: int
    The number of samples in a data batch.

  time_steps: int
    The number of time steps (or the context length) of the input sequence.

  n_features: int
    The number of features in the time series.

  dropout: float, default: 0
    Fraction of the units to drop for the linear transformation.

  recurrent_dropout: float, default: 0
    Fraction of the units to drop for the linear transformation of the recurrent state.

  Returns
  -------
  A LSTM model, an instance of tf.keras.Sequential.
  """
  return tf.keras.Sequential([
    tf.keras.layers.LSTM(units=lstm_units,
                         stateful=True,
                         batch_input_shape=(batch_size, time_steps, n_features),
                         dropout=dropout,
                         recurrent_dropout=recurrent_dropout,
                         return_sequences=True),
    tf.keras.layers.Dropout(dropout),
    tf.keras.layers.LSTM(units=lstm_units,
                         dropout=dropout,
                         recurrent_dropout=recurrent_dropout,
                         stateful=True),
    tf.keras.layers.Dropout(dropout),
    tf.keras.layers.Dense(output_size)
  ])

def create_stateless_lstm_model(lstm_units,
                                output_size,
                                time_steps,
                                n_features,
                                dropout=0.,
                                recurrent_dropout=0.):
  """
  Create a LSTM model.

  Parameters
  ----------
  lstm_units: int
    The number of units of LSTM's output space.

  output_size: int
    The number of units in final dense layer.

  batch_size: int
    The number of samples in a data batch.

  time_steps: int
    The number of time steps (or the context length) of the input sequence.

  n_features: int
    The number of features in the time series.

  dropout: float, default: 0
    Fraction of the units to drop for the linear transformation.

  recurrent_dropout: float, default: 0
    Fraction of the units to drop for the linear transformation of the recurrent state.

  Returns
  -------
  A LSTM model, an instance of tf.keras.Sequential.
  """
  return tf.keras.Sequential([
    tf.keras.layers.LSTM(units=lstm_units,
                         input_shape=(time_steps, n_features),
                         dropout=dropout,
                         recurrent_dropout=recurrent_dropout,
                         return_sequences=True),
    tf.keras.layers.Dropout(dropout),
    tf.keras.layers.LSTM(units=lstm_units,
                         dropout=dropout,
                         recurrent_dropout=recurrent_dropout),
    tf.keras.layers.Dropout(dropout),
    tf.keras.layers.Dense(output_size)
  ])