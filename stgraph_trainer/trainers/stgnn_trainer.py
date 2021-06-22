import torch
import numpy as np
import math
import time
from .base import BaseTrainer

class STGNNTrainer(BaseTrainer):
  """
  Trainer class for the STGNN model.

  Parameters
  ----------
  model: object
    The model to train and predict.

  train_ds: object
    The train dataloader.

  test_ds: object for STGNN, or tensor for Proposed STGNN
    shape of (n_test_samples, n_nodes, time_steps, in_channels)
    The test dataloader for STGNN, full test tensor for Proposed STGNN.

  adj: 2D tensor shape of (n_nodes, n_nodes)
    Adjacency matrix for graph convolution.

  scaler: object
    Scaler object used to inverse predictions back to original scale.

  loss_func: object
    The loss object function.

  optimizer: object
    The optimizer used to train the model.

  device: torch.cuda object
    The device to execute.

  raw_test: array of shape (n_test_samples + 1, n_features)
    The raw test dataset used for computing metrics.
    Here, n_features is the number of predicted time series (provinces).
    1 upper row is appended to inverse differencing.

  Attributes
  ----------
  history: dict of list
    Information of training. None when initialize.
  """
  def __init__(self,
               model,
               train_ds,
               test_ds,
               adj,
               scaler,
               loss_func,
               optimizer,
               device,
               raw_test=None,
               *args,
               **kwargs):
    super(STGNNTrainer, self).__init__(model,
                                       train_ds,
                                       test_ds,
                                       loss_func,
                                       optimizer,
                                       callbacks=None)
    if raw_test is None:
      raise Exception('No raw test data.')
    self.raw_test = raw_test
    self.scaler = scaler
    self.device = device
    self.adj = adj.to(device)

  def train(self, epochs):
    """
    Train the model.

    Parameters
    ----------
    epochs: int
      The number of training epochs.

    Returns
    -------
    history: dict of list
      Information of training.
    """
    self.history = {'epoch': [],
                    'train_loss': [],
                    'test_loss': [],
                    'elapsed_time': []}
    
    for epoch in range(epochs):
      total_loss = 0.
      time_steps = self.model.time_steps
      n_steps_per_epoch = len(self.train_ds)
      
      start_epoch_train_time = time.time()
      for x_batch, y_batch in self.train_ds:
        x_batch = x_batch.squeeze(0).to(self.device)
        y_batch = y_batch.T.to(self.device)
        loss = self.train_step(x_batch, y_batch, self.adj)
        total_loss += loss.item()
      end_epoch_train_time = time.time()

      t_loss = self.evaluate()

      self.history['epoch'].append(epoch + 1)
      self.history['train_loss'].append(total_loss / n_steps_per_epoch)
      self.history['test_loss'].append(t_loss.item())
      self.history['elapsed_time'].append(end_epoch_train_time - start_epoch_train_time)

      msg = 'Epoch: {}; Elapsed time: {}; Train loss: {:.6f}; Test MSE: {:.6f}; Test loss RMSE: {:.6f}'
      print(msg.format(self.history['epoch'][-1],
                       self.history['elapsed_time'][-1],
                       self.history['train_loss'][-1],
                       self.history['test_loss'][-1],
                       math.sqrt(self.history['test_loss'][-1])))

    return self.history

  def train_step(self, x_batch, y_batch, adj):
    """
    Train the model for 1 step (or batch).

    Parameters
    ----------
    x_batch: tensor
      Input batch data for features.

    y_batch: tensor
      Input batch data for labels.

    Returns
    -------
    loss: float Tensor
      Training loss of a batch.
    """
    self.model.train()
    self.optimizer.zero_grad()
    y_pred = self.model(x_batch, adj)
    loss = self.loss_func(y_pred, y_batch)
    loss.backward()
    self.optimizer.step()
    return loss.detach()

  def evaluate(self):
    """
    Conduct prediction and evaluation.

    Returns
    -------
    metrics: float Tensor
      The loss on the raw test dataset.
    """
    predictions = self.predict()
    with torch.no_grad():
      loss = self.loss_func(torch.tensor(predictions), torch.tensor(self.raw_test[1:]))
    return loss.detach()

  def predict(self):
    """
    Make prediction.

    Returns
    -------
    predictions: array of shape (n_samples, n_features)
      The prediction used for computing metrics.
      Here, n_features is the number of predicted time series (provinces).
    """
    predictions = []
    self.model.eval()
    for i, (x_batch, y_batch) in enumerate(self.test_ds):
      with torch.no_grad():
        x_batch = x_batch.squeeze(0).to(self.device)
        y_pred = self.model(x_batch, self.adj)
        
        # Inverse prediction to original scale
        y_pred = y_pred.cpu().numpy().T
        y_pred = self.scaler.inverse_transform(y_pred)
        y_pred = self.raw_test[i, :] + y_pred
        predictions.append(y_pred.copy())
    return np.concatenate(predictions, axis=0)

class ModifiedSTGNNTrainer(BaseTrainer):
  """
  Trainer class for the STGNN model.

  Parameters
  ----------
  model: object
    The model to train and predict.

  train_ds: object
    The train dataloader.

  test_ds: object for STGNN, or tensor for Proposed STGNN
    shape of (n_test_samples, n_nodes, time_steps, in_channels)
    The test dataloader for STGNN, full test tensor for Proposed STGNN.

  adj: 2D tensor shape of (n_nodes, n_nodes)
    Adjacency matrix for graph convolution.

  scaler: object
    Scaler object used to inverse predictions back to original scale.

  loss_func: object
    The loss object function.

  optimizer: object
    The optimizer used to train the model.

  device: torch.cuda object
    The device to execute.

  raw_test: array of shape (n_test_samples + 1, n_features)
    The raw test dataset used for computing metrics.
    Here, n_features is the number of predicted time series (provinces).
    1 upper row is appended to inverse differencing.

  Attributes
  ----------
  history: dict of list
    Information of training. None when initialize.
  """
  def __init__(self,
               model,
               train_ds,
               test_ds,
               adj,
               scaler,
               loss_func,
               optimizer,
               device,
               raw_test=None,
               *args,
               **kwargs):
    super(ModifiedSTGNNTrainer, self).__init__(model,
                                               train_ds,
                                               test_ds,
                                               loss_func,
                                               optimizer,
                                               callbacks=None)
    if raw_test is None:
      raise Exception('No raw test data.')
    self.raw_test = raw_test
    self.scaler = scaler
    self.device = device
    self.adj = adj.to(device)

  def train(self, epochs):
    """
    Train the model.

    Parameters
    ----------
    epochs: int
      The number of training epochs.

    Returns
    -------
    history: dict of list
      Information of training.
    """
    self.history = {'epoch': [],
                    'train_loss': [],
                    'test_loss': [],
                    'elapsed_time': []}
    
    for epoch in range(epochs):
      total_loss = 0.
      epoch_training_losses = []
      # time_steps = self.model.time_steps
      # n_steps_per_epoch = len(self.train_ds)
      
      start_epoch_train_time = time.time()
      for x_batch, y_batch in self.train_ds:
        x_batch = x_batch.to(self.device)
        y_batch = y_batch.to(self.device)
        loss = self.train_step(x_batch, y_batch, self.adj)
        # total_loss += loss.item()
        epoch_training_losses.append(loss.item())
      end_epoch_train_time = time.time()

      t_loss = self.evaluate()

      self.history['epoch'].append(epoch + 1)
      self.history['train_loss'].append(sum(epoch_training_losses)/len(epoch_training_losses))
      self.history['test_loss'].append(t_loss.item())
      self.history['elapsed_time'].append(end_epoch_train_time - start_epoch_train_time)

      msg = 'Epoch: {}; Elapsed time: {}; Train loss: {:.6f}; Test MSE: {:.6f}; Test loss RMSE: {:.6f}'
      print(msg.format(self.history['epoch'][-1],
                       self.history['elapsed_time'][-1],
                       self.history['train_loss'][-1],
                       self.history['test_loss'][-1],
                       math.sqrt(self.history['test_loss'][-1])))

    return self.history

  def train_step(self, x_batch, y_batch, adj):
    """
    Train the model for 1 step (or batch).

    Parameters
    ----------
    x_batch: tensor
      Input batch data for features.

    y_batch: tensor
      Input batch data for labels.

    Returns
    -------
    loss: float Tensor
      Training loss of a batch.
    """
    self.model.train()
    self.optimizer.zero_grad()
    y_pred = self.model(x_batch, adj)
    # print(y_pred.shape, y_batch.shape)
    loss = self.loss_func(y_pred, y_batch)
    loss.backward()
    self.optimizer.step()
    return loss.detach()

  def evaluate(self):
    """
    Conduct prediction and evaluation.

    Returns
    -------
    metrics: float Tensor
      The loss on the raw test dataset.
    """
    predictions = self.predict()
    with torch.no_grad():
      loss = self.loss_func(torch.tensor(predictions), torch.tensor(self.raw_test[1:]))
    return loss.detach()

  def predict(self):
    """
    Make prediction.

    Returns
    -------
    predictions: array of shape (n_samples, n_features)
      The prediction used for computing metrics.
      Here, n_features is the number of predicted time series (provinces).
    """
    predictions = []
    self.model.eval()
    for i, (x_batch, y_batch) in enumerate(self.test_ds):
      with torch.no_grad():
        x_batch = x_batch.to(self.device)
        y_pred = self.model(x_batch, self.adj)
        
        # Inverse prediction to original scale
        y_pred = y_pred.cpu().numpy()
        y_pred = self.scaler.inverse_transform(y_pred)
        y_pred = self.raw_test[i, :] + y_pred
        predictions.append(y_pred.copy())
    return np.concatenate(predictions, axis=0)

class ProposedSTGNNTrainer(STGNNTrainer):
  def train(self, epochs):
    """
    Train the model.

    Parameters
    ----------
    epochs: int
      The number of training epochs.

    Returns
    -------
    history: dict of list
      Information of training.
    """
    self.history = {'epoch': [],
                    'train_loss': [],
                    'test_loss': [],
                    'elapsed_time': []}
    
    for epoch in range(epochs):
      epoch_training_losses = []
      time_steps = self.model.time_steps
      
      start_epoch_train_time = time.time()
      for x_batch, y_batch in self.train_ds:
        x_batch = x_batch.to(self.device)
        y_batch = y_batch.to(self.device)
        loss = self.train_step(x_batch, y_batch, self.adj)
        epoch_training_losses.append(loss.detach().cpu().numpy())
      
      end_epoch_train_time = time.time()

      t_loss = self.evaluate()

      self.history['epoch'].append(epoch + 1)
      self.history['train_loss'].append(sum(epoch_training_losses)/len(epoch_training_losses))
      self.history['test_loss'].append(t_loss.item())
      self.history['elapsed_time'].append(end_epoch_train_time - start_epoch_train_time)

      msg = 'Epoch: {}; Elapsed time: {}; Train loss: {:.6f}; Test MSE: {:.6f}; Test loss RMSE: {:.6f}'
      print(msg.format(self.history['epoch'][-1],
                       self.history['elapsed_time'][-1],
                       self.history['train_loss'][-1],
                       self.history['test_loss'][-1],
                       math.sqrt(self.history['test_loss'][-1])))

    return self.history
  
  def predict(self):
    """
    Make prediction.

    Returns
    -------
    predictions: array of shape (n_samples, n_features)
      The prediction used for computing metrics.
      Here, n_features is the number of predicted time series (provinces).
    """
    with torch.no_grad():
      self.model.eval()
      val_input = self.test_ds.to(self.device)
      predictions = self.model(val_input, self.adj)
      # Inverse prediction to original scale
      predictions = predictions.squeeze().detach().cpu().numpy()
      predictions = self.scaler.inverse_transform(predictions)
      predictions = predictions + self.raw_test[:-1]
    return predictions