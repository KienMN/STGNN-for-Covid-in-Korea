# Import libraries
from stgraph_trainer.datasets import load_province_temporal_data
from stgraph_trainer.datasets import data_diff
from stgraph_trainer.datasets import timeseries_to_supervised
from stgraph_trainer.datasets import load_province_coordinates
from stgraph_trainer.datasets import preprocess_data_for_stgnn
from stgraph_trainer.datasets import inverse_diff
from stgraph_trainer.utils import PairDataset
from stgraph_trainer.utils import get_config_from_json
from stgraph_trainer.utils import compute_metrics
from stgraph_trainer.utils import save_predictions
from stgraph_trainer.utils import save_metrics
from stgraph_trainer.utils import get_distance_in_km_between_earth_coordinates
from stgraph_trainer.utils import get_adjacency_matrix
from stgraph_trainer.utils import get_normalized_adj
from torch.utils.data import DataLoader
from stgraph_trainer.models import STGNN
from stgraph_trainer.trainers import STGNNTrainer
# from stgraph_trainer.callbacks import PostPredictionCallback
import os
import torch
# import tensorflow as tf
import numpy as np
from functools import partial

# Set up configs and parameters
data_config_file = os.path.dirname(__file__) + '/configs/data_config.json'
model_config_file = os.path.dirname(__file__) + '/configs/stgnn_config.json'
model_name = 'stgnn'
work_dir = os.path.abspath(os.path.dirname(__file__)) + '/results'

data_configs = get_config_from_json(data_config_file)
model_configs = get_config_from_json(model_config_file)

PROVINCES = data_configs['provinces']
SPLIT_DATE = data_configs['split_date']
TIME_STEPS = int(data_configs['time_steps'])
STATUS = data_configs['status']

TEMP_FEAT = int(model_configs['temp_feat'])
IN_FEAT = int(model_configs['in_feat'])
HIDDEN_FEAT = int(model_configs['hidden_feat'])
OUT_FEAT = int(model_configs['out_feat'])
PRED_FEAT = int(model_configs['pred_feat'])
BIAS = bool(model_configs['bias'])
DROP_RATE = float(model_configs['drop_rate'])

TRIALS = int(model_configs['trials'])
EPOCHS = int(model_configs['epochs'])
BATCH_SIZE = int(model_configs['batch_size'])

if torch.cuda.is_available():
  device = torch.device('cuda', 0)
else:
  device = torch.device('cpu')
print(device)

# Load and process dataset
df = load_province_temporal_data(provinces=PROVINCES, status=STATUS)

X_train, y_train, X_test, y_test, _, _, scaler = preprocess_data_for_stgnn(df,
                                                                           SPLIT_DATE,
                                                                           TIME_STEPS)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
n_test_samples = len(y_test)
print(n_test_samples)

# Coordinates data
province_coords = load_province_coordinates().values[:, 1:]

dist_km = []
for idx, c1 in enumerate(province_coords):
  dist_km.append([get_distance_in_km_between_earth_coordinates(c1, c2) for c2 in province_coords])
dist_mx = np.array(dist_km)

adj_mx = get_adjacency_matrix(dist_mx)
# Fix formatting
adj_mx = adj_mx.astype(np.float32)

adj_mx = get_normalized_adj(adj_mx)
adj = torch.tensor(adj_mx)
# print(adj)

# Create model, train and evaluate
rmse_results = []
mae_results = []

for trial in range(TRIALS):
  model = STGNN(TEMP_FEAT,
                IN_FEAT,
                HIDDEN_FEAT,
                OUT_FEAT,
                PRED_FEAT,
                DROP_RATE,
                BIAS)
  
  model.to(device)
  # print(model)

  train_dl = DataLoader(PairDataset(X_train, y_train),
                        batch_size=BATCH_SIZE,
                        shuffle=False)

  test_dl = DataLoader(PairDataset(X_test, y_test),
                        batch_size=BATCH_SIZE,
                        shuffle=False)

  loss_func = torch.nn.MSELoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

  trainer = STGNNTrainer(model,
                         train_dl,
                         test_dl,
                         adj,
                         scaler,
                         loss_func,
                         optimizer,
                         device,
                         callbacks=None,
                         raw_test=df.iloc[-(n_test_samples + 1):].values)

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