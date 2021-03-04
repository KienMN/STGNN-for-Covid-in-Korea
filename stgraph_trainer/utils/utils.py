from typing import *
import tensorflow as tf
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import math

class PairDataset(Dataset):
  """
  A PyTorch Dataset subclass to store (input, label) pair

  Parameters
  ----------
  inputs: iterable object
    The input dataset.

  labels: iterable object
    The label for input accordingly.
  """
  def __init__(self, inputs, labels):
    self.inputs = inputs
    self.labels = labels

  def __len__(self):
    """Length of the dataset."""
    return len(self.labels)

  def __getitem__(self, idx):
    """Get item of dataset, given index idx."""
    return self.inputs[idx], self.labels[idx]

def listify(o):
  """Make a list from the input object."""
  if o is None:
    return []
  if isinstance(o, list):
    return o
  if isinstance(o, str):
    return [o]
  if isinstance(o, Iterable):
    return list(o)
  return [o]

def compose(x, funcs, *args, order_key='_order', **kwargs):
  """Apply a list of funcs to input x in ascending order of order_key."""
  key = lambda o: getattr(o, order_key, 0)
  for f in sorted(listify(funcs), key=key):
    x = f(x, **kwargs)
  return x

def compute_metrics(true_df, pred_df, metric='rmse'):
  """
  Compute metrics for true values and predictions.

  Parameters
  ----------
  true_df: 2D array or DataFrame shape of (n_samples, n_columns)
    True array values.

  pred_df: 2D array or DataFrame shape of (n_samples, n_columns)
    Prediction values.

  metric: str, options: ['rmse', 'mae', 'mape'], default: 'rmse'
    Metric name to compute.

  Returns
  -------
  output: array shape of (n_columns,)
    Metric values for each column in the array.

  output_avg: float
    Average value of the metric, element-wise.
  """
  if isinstance(true_df, pd.DataFrame):
    true_df = true_df.values
  if isinstance(pred_df, pd.DataFrame):
    pred_df = pred_df.values

  if metric == 'rmse':
    f = tf.metrics.mean_squared_error
  elif metric == 'mae':
    f = tf.metrics.mean_absolute_error
  elif metric == 'mape':
    f = tf.metrics.mean_absolute_percentage_error

  output = []
  n = len(true_df)

  for i in range(true_df.shape[1]):
    y_true = true_df[:, i]
    y_pred = pred_df[:, i]
    m = f(y_true, y_pred).numpy()
    output.append(m)
  
  output = np.array(output)
  output_avg = (output * n).sum() / (n * len(output))

  if metric == 'rmse':
    output = np.sqrt(output)
    output_avg = math.sqrt(output_avg)
  
  return output, output_avg

def save_predictions(y_pred,
                     model_name,
                     n_exp,
                     columns=None,
                     index=None,
                     path=''):
  """Save predictions according to model_name, n_exp in path directory."""
  df = pd.DataFrame(y_pred,
                    columns=columns,
                    index=index)
  df.to_csv(path + '/{}_pred_{}.csv'.format(model_name, str(n_exp)),
            index=index is not None)

def save_metrics(metrics, model_name, metric_name, columns=None, path=''):
  """Save metrics according to model_name, metric_name in path directory."""
  df = pd.DataFrame(metrics, columns=columns)
  df.to_csv(path + '/{}_{}.csv'.format(model_name, metric_name),
            index=False)

def get_distance_in_km_between_earth_coordinates(c1, c2):
  """Compute distance in km between 2 coordinates."""
  lat1, lon1 = c1
  lat2, lon2 = c2
  dLat = np.radians(lat2-lat1)
  dLon = np.radians(lon2-lon1)
  lat1 = np.radians(lat1)
  lat2 = np.radians(lat2)
  a = np.sin(dLat/2) * np.sin(dLat/2) + np.sin(dLon/2) * np.sin(dLon/2) * np.cos(lat1) * np.cos(lat2)
  c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
  earth_radius_km = 6371
  return earth_radius_km * c

def get_adjacency_matrix(dist_mx, normalized_k=0.1):
  """
  Compute adjacency matrix for a distance matrix.

  Parameters
  ---------- 
  dist_mx: 2D array shape of (n_entries, n_entries)
    A distance matrix.
  
  normalized_k: float, default: 0.1
    Entries that become lower than normalized_k after normalization
    are set to zero for sparsity.
  
  Returns
  -------
  adj_mx: 2D array shape of (n_entries)
    Adjacency matrix for the distance matrix.
  """

  # Calculates the standard deviation as theta.
  distances = dist_mx[~np.isinf(dist_mx)].flatten()
  std = distances.std()
  adj_mx = np.exp(-np.square(dist_mx / std))
  # Make the adjacent matrix symmetric by taking the max.
  # adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])

  # Sets entries that lower than a threshold, i.e., k, to zero for sparsity.
  adj_mx[adj_mx < normalized_k] = 0
  return adj_mx

def get_normalized_adj(A):
  """
  Returns the degree normalized adjacency matrix.
  """
  A = A + np.diag(np.ones(A.shape[0], dtype=np.float32))
  D = np.array(np.sum(A, axis=1)).reshape((-1,))
  D[D <= 10e-5] = 10e-5    # Prevent infs
  diag = np.reciprocal(np.sqrt(D))
  A_wave = np.multiply(np.multiply(diag.reshape((-1, 1)), A),
                        diag.reshape((1, -1)))
  return A_wave

# def np_compute_metrics(true_arr, pred_arr, metric='rmse'):
#   mae = np.abs(np.subtract(true_arr, pred_arr))

#   if metric == 'mae':
#     return np.mean(mae)
#   if metric == 'rmse':
#     mse = np.square(mae)
#     return np.sqrt(np.mean(mse))