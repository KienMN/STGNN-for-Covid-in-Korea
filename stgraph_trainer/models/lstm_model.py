import tensorflow as tf

def create_lstm_model(lstm_units,
                      output_size,
                      batch_size,
                      time_steps,
                      n_features,
                      drop_rate=0.5,
                      recurrent_drop_rate=0.5):
  return tf.keras.Sequential([
    tf.keras.layers.LSTM(units=lstm_units,
                         stateful=True,
                         batch_input_shape=(batch_size, time_steps, n_features),
                         dropout=drop_rate,
                         recurrent_dropout=recurrent_drop_rate,
                         return_sequences=True),
    tf.keras.layers.Dropout(drop_rate),
    tf.keras.layers.LSTM(units=lstm_units,
                         dropout=drop_rate,
                         recurrent_dropout=recurrent_drop_rate,
                         stateful=True),
    tf.keras.layers.Dropout(drop_rate),
    tf.keras.layers.Dense(output_size)
  ])