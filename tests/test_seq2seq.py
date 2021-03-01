# Import libraries
from stgraph_trainer.datasets import load_province_temporal_data
from stgraph_trainer.datasets import data_diff
from stgraph_trainer.datasets import timeseries_to_supervised
from stgraph_trainer.datasets import preprocess_data_for_seq2seq
from stgraph_trainer.datasets import inverse_diff
from stgraph_trainer.utils import get_config_from_json
from stgraph_trainer.utils import compute_metrics
from stgraph_trainer.utils import save_predictions
from stgraph_trainer.utils import save_metrics
from stgraph_trainer.models import Encoder, Decoder, DecoderWithAttention
from stgraph_trainer.trainers import Seq2SeqTrainer
from stgraph_trainer.callbacks import PostPredictionCallback
import os
import tensorflow as tf
import numpy as np
from functools import partial

# Set up configs and parameters
data_config_file = os.path.dirname(__file__) + '/configs/data_config.json'
seq2seq_config_file = os.path.dirname(__file__) + '/configs/seq2seq_config.json'
model_name = 'seq2seq'
work_dir = os.path.abspath(os.path.dirname(__file__)) + '/results'

data_configs = get_config_from_json(data_config_file)
seq2seq_configs = get_config_from_json(seq2seq_config_file)

PROVINCES = data_configs['provinces']
SPLIT_DATE = data_configs['split_date']
TIME_STEPS = int(data_configs['time_steps'])
STATUS = data_configs['status']

ENC_DENSE_UNITS = int(seq2seq_configs['enc_dense_units'])
ENC_UNITS = int(seq2seq_configs['enc_units'])
DEC_UNITS = int(seq2seq_configs['dec_units'])
BATCH_SIZE = int(seq2seq_configs['batch_size'])
DROP_RATE = float(seq2seq_configs['drop_rate'])
RECURRENT_DROP_RATE = float(seq2seq_configs['recurrent_drop_rate'])
TRIALS = int(seq2seq_configs['trials'])
EPOCHS = int(seq2seq_configs['epochs'])

# Load and process dataset
df = load_province_temporal_data(provinces=PROVINCES, status=STATUS)

X_train, y_train, X_test, y_test, _, raw_test, scaler = preprocess_data_for_seq2seq(df,
                                                                                    SPLIT_DATE,
                                                                                    TIME_STEPS)

n_features = X_train.shape[-1]
n_test_samples = len(raw_test) - TIME_STEPS
output_size = X_train.shape[-1]

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

# Functions to handle post-prediction
# including inverse transform and add back difference
# to make predictions back to original scale.
def inverse_transform(x, scaler, **kwargs):
  original_shape = x.shape
  return scaler.inverse_transform(x.reshape((-1, 1))).reshape(original_shape)
inv_trans = partial(inverse_transform, scaler=scaler)
inv_trans._order = 10

def inv_diff_1(x, raw_values):
  return x + raw_values
inv_diff = partial(inv_diff_1, raw_values=df.iloc[-(n_test_samples + 1):-1].values)
inv_diff._order = 20

tfms = [inv_trans, inv_diff]

# Create model, train and evaluate
rmse_results = []
mae_results = []

for trial in range(TRIALS):
  encoder = Encoder(ENC_DENSE_UNITS,
                    ENC_UNITS,
                    BATCH_SIZE,
                    dropout=DROP_RATE,
                    recurrent_dropout=RECURRENT_DROP_RATE)

  # decoder = Decoder(1,
  #                   DEC_UNITS,
  #                   BATCH_SIZE,
  #                   dropout=DROP_RATE,
  #                   recurrent_dropout=RECURRENT_DROP_RATE)

  decoder = DecoderWithAttention(1,
                                 DEC_UNITS,
                                 BATCH_SIZE,
                                 dropout=DROP_RATE,
                                 recurrent_dropout=RECURRENT_DROP_RATE)
  
  model = [encoder, decoder]

  train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)) \
                            .shuffle(10000) \
                            .batch(BATCH_SIZE, drop_remainder=True)
  test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(BATCH_SIZE * 2)

  loss_func = tf.losses.MeanSquaredError()
  optimizer = tf.optimizers.Adam(learning_rate=0.01)

  trainer = Seq2SeqTrainer(model,
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

# Save metrics
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