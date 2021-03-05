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
from stgraph_trainer.models import ProposedSTGNN
from stgraph_trainer.trainers import STGNNTrainer, ProposedSTGNNTrainer
import os
import torch
import numpy as np

# Set up configs and parameters
data_config_file = os.path.dirname(__file__) + '/configs/data_config.json'
model_config_file = os.path.dirname(__file__) + '/configs/proposed_stgnn_config.json'
model_name = "proposed_stgnn"
work_dir = os.path.abspath(os.path.dirname(__file__)) + '/results'

data_configs = get_config_from_json(data_config_file)
model_configs = get_config_from_json(model_config_file)

PROVINCES = data_configs['provinces']
SPLIT_DATE = data_configs['split_date']
TIME_STEPS = int(data_configs['time_steps'])
STATUS = data_configs['status']

N_NODES = len(PROVINCES)
PREDICTED_TIME_STEPS = int(model_configs['predicted_time_steps'])
SPATIAL_CHANNELS = int(model_configs['spatial_channels'])
SPATIAL_HIDDEN_CHANNELS = int(model_configs['spatial_hidden_channels'])
SPATIAL_OUT_CHANNELS = int(model_configs['spatial_out_channels'])
OUT_CHANNELS = int(model_configs['out_channels'])
TEMPORAL_KERNEL = int(model_configs['temporal_kernel'])
BATCH_NORM = bool(model_configs['batch_norm'])
DROP_RATE = float(model_configs['drop_rate'])

TRIALS = int(model_configs['trials'])
EPOCHS = int(model_configs['epochs'])
BATCH_SIZE = int(model_configs['batch_size'])

if torch.cuda.is_available():
  device = torch.device('cuda', 0)
else:
  device = torch.device('cpu')

# Load and process dataset
df = load_province_temporal_data(provinces=PROVINCES, status=STATUS)

X_train, y_train, X_test, y_test, _, _, scaler = preprocess_data_for_stgnn(df,
                                                                           SPLIT_DATE,
                                                                           TIME_STEPS)

X_train = torch.tensor(X_train).unsqueeze(-1)
y_train = torch.tensor(y_train).unsqueeze(-1)
X_test = torch.tensor(X_test).unsqueeze(-1)
y_test = torch.tensor(y_test).unsqueeze(-1)

n_test_samples = len(y_test)
IN_CHANNELS = X_train.shape[-1]
# print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
# print(n_test_samples)

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
  model = ProposedSTGNN(N_NODES,
                        TIME_STEPS,
                        PREDICTED_TIME_STEPS,
                        IN_CHANNELS,
                        SPATIAL_CHANNELS,
                        SPATIAL_HIDDEN_CHANNELS,
                        SPATIAL_OUT_CHANNELS,
                        OUT_CHANNELS,
                        TEMPORAL_KERNEL,
                        DROP_RATE,
                        BATCH_NORM)
  
  model.to(device)
  # print(model)

  train_dl = DataLoader(PairDataset(X_train, y_train),
                        batch_size=BATCH_SIZE,
                        shuffle=True)

  loss_func = torch.nn.MSELoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

  trainer = ProposedSTGNNTrainer(model,
                                 train_dl,
                                 X_test,
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
  # print(predict.shape)

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