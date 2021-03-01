from typing import *
import tensorflow as tf
import pandas as pd
import numpy as np
import math

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