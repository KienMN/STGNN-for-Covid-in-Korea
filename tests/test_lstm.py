from stgraph_trainer.datasets import load_province_temporal_data
from stgraph_trainer.datasets import data_diff
from stgraph_trainer.datasets import timeseries_to_supervised
from stgraph_trainer.datasets import preprocess_data_for_lstm_model
from stgraph_trainer.datasets import inverse_diff
from stgraph_trainer.utils import get_config_from_json
from stgraph_trainer.utils import compute_metrics
from stgraph_trainer.utils import save_predictions
from stgraph_trainer.utils import save_metrics
from sklearn.preprocessing import MinMaxScaler
from stgraph_trainer.models import create_lstm_model
from stgraph_trainer.trainers import LSTMTrainer
from stgraph_trainer.callbacks import PostPredictionCallback
import os
import tensorflow as tf
import numpy as np
from functools import partial

data_config_file = os.path.dirname(__file__) + '/configs/data_config.json'
lstm_config_file = os.path.dirname(__file__) + '/configs/lstm_config.json'
model_name = 'lstm'
work_dir = os.path.abspath(os.path.dirname(__file__)) + '/results/'

data_configs = get_config_from_json(data_config_file)
lstm_configs = get_config_from_json(lstm_config_file)

PROVINCES = data_configs['provinces']
SPLIT_DATE = data_configs['split_date']
TIME_STEPS = int(data_configs['time_steps'])
STATUS = data_configs['status']

LSTM_UNITS = int(lstm_configs['lstm_units'])
BATCH_SIZE = int(lstm_configs['batch_size'])
DROP_RATE = float(lstm_configs['drop_rate'])
RECURRENT_DROP_RATE = float(lstm_configs['recurrent_drop_rate'])
TRIALS = int(lstm_configs['trials'])
EPOCHS = int(lstm_configs['epochs'])

df = load_province_temporal_data(provinces=PROVINCES, status=STATUS)

X_train, y_train, X_test, y_test, raw_train, raw_test, scaler = preprocess_data_for_lstm_model(df, SPLIT_DATE, TIME_STEPS)

n_features = X_train.shape[-1]
n_test_samples = len(y_test)
output_size = X_train.shape[-1]

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

def inverse_transform(x, scaler, **kwargs):
  return scaler.inverse_transform(x)

inv_trans = partial(inverse_transform, scaler=scaler)
inv_trans._order = 10

def inv_diff_1(x, raw_values, n_test=None, idx=0):
  return inverse_diff(x, raw_values, n_test + 1 - idx)
inv_diff = partial(inv_diff_1, raw_values=df.values, n_test=n_test_samples)
inv_diff._order = 20

tfms = [inv_trans, inv_diff]

rmse_results = []
mae_results = []

for trial in range(TRIALS):
  model = create_lstm_model(LSTM_UNITS,
                            output_size,
                            BATCH_SIZE,
                            TIME_STEPS,
                            n_features,
                            DROP_RATE,
                            RECURRENT_DROP_RATE)

  print(model.summary())

  train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(BATCH_SIZE)
  test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(BATCH_SIZE)

  loss_func = tf.losses.MeanSquaredError()
  optimizer = tf.optimizers.Adam(learning_rate=0.01)

  trainer = LSTMTrainer(model,
                        train_ds,
                        test_ds,
                        loss_func,
                        optimizer,
                        callbacks=[PostPredictionCallback(funcs=tfms)],
                        raw_test=df.iloc[-n_test_samples:].values)

  history = trainer.train(EPOCHS)
  print(history)

  predict = trainer.predict()
  save_predictions(predict,
                   model_name,
                   n_exp=trial,
                   columns=PROVINCES,
                   index=df.iloc[-n_test_samples:].index,
                   path=work_dir)
  print(predict.shape)

  m, m_avg = compute_metrics(df.iloc[-n_test_samples:], predict, metric='rmse')
  m = np.append(m, m_avg)
  rmse_results.append(m)
  print(m, m_avg)
  
  m, m_avg = compute_metrics(df.iloc[-n_test_samples:], predict, metric='mae')
  m = np.append(m, m_avg)
  mae_results.append(m)
  print(m, m_avg)

save_metrics(rmse_results,
             columns=PROVINCES + ['Avg'],
             model_name=model_name,
             metric_name='rmse',
             path=work_dir)

save_metrics(mae_results,
             columns=PROVINCES + ['Avg'],
             model_name=model_name,
             metric_name='mae',
             path=work_dir)