import tensorflow as tf
import numpy as np
import math
import time
from .base import BaseTrainer

class LSTMTrainer(BaseTrainer):
  """
  Trainer class for the LSTM model.

  Parameters
  ----------
  model: object
    The model to train and predict.

  train_ds: object
    The train dataset.

  test_ds: object
    The test dataset.

  loss_func: object
    The loss object function.

  optimizer: object
    The optimizer used to train the model.

  callbacks: list or objects, default None
    The list of callbacks to execute.

  raw_test: array of shape (n_samples, n_features)
    The raw test dataset used for computing metrics.
    Here, n_features is the number of predicted time series (provinces).

  Attributes
  ----------
  history: dict of list
    Information of training. None when initialize.

  train_loss: object
    Metric object to compute the loss of training.

  test_loss: object
    Metric object to compute the loss of testing.
  """
  def __init__(self,
               model,
               train_ds,
               test_ds,
               loss_func,
               optimizer,
               callbacks=None,
               raw_test=None,
               *args,
               **kwargs):
    super(LSTMTrainer, self).__init__(model,
                                      train_ds,
                                      test_ds,
                                      loss_func,
                                      optimizer,
                                      callbacks)
    if raw_test is None:
      raise Exception('No raw test data.')
    self.raw_test = raw_test
    self.train_loss = tf.keras.metrics.Mean()
    self.test_loss = tf.keras.metrics.Mean()

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
      self.train_loss.reset_states()
      self.test_loss.reset_states()
      self.model.reset_states()

      start_epoch_train_time = time.time()
      for x_batch, y_batch in self.train_ds:
        loss = self.train_step(x_batch, y_batch)
        self.train_loss(loss)
      end_epoch_train_time = time.time()

      t_loss = self.evaluate()

      self.history['epoch'].append(epoch + 1)
      self.history['train_loss'].append(self.train_loss.result().numpy())
      self.history['test_loss'].append(t_loss.numpy())
      self.history['elapsed_time'].append(end_epoch_train_time - start_epoch_train_time)

      msg = 'Epoch: {}; Elapsed time: {}; Train loss: {:.6f}; Test MSE: {:.6f}; Test loss RMSE: {:.6f}'
      print(msg.format(self.history['epoch'][-1],
                       self.history['elapsed_time'][-1],
                       self.history['train_loss'][-1],
                       self.history['test_loss'][-1],
                       math.sqrt(self.history['test_loss'][-1])))

    return self.history

  def train_step(self, x_batch, y_batch):
    """
    Train the model for 1 step (or batch).

    Parameters
    ----------
    x_batch: array shape of (batch_size, time_steps, n_features)
      Input batch data for features.

    y_batch: array shape of (batch_size, output_size)
      Input batch data for labels, with output_size is the number of features/nodes.

    Returns
    -------
    loss: float Tensor
      Training loss of a batch.
    """
    with tf.GradientTape() as tape:
      y_pred = self.model(x_batch, training=True)
      assert y_pred.shape == y_batch.shape
      loss = self.loss_func(y_true=y_batch, y_pred=y_pred)
    gradients = tape.gradient(loss, self.model.trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
    return loss

  def evaluate(self):
    """
    Conduct prediction and evaluation.

    Returns
    -------
    metrics: float Tensor
      The loss on the raw test dataset.
    """
    predictions = self.predict()
    return self.loss_func(self.raw_test, predictions)

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
    for i, (x_batch, y_batch) in enumerate(self.test_ds):
      self.y_hat = self.model(x_batch, training=False)
      self.idx = i

      # Place for callback to inverse results
      self.callbacks.on_test_batch_end(batch=i)
      predictions.append(self.y_hat.copy())
    return np.concatenate(predictions, axis=0)

class LSTMOneTrainer(LSTMTrainer):
  """
  Trainer class for the LSTM model
  with input shape = (batch_size, n_timesteps, 1).
  """
  def predict(self):
    predictions = np.array([])
    for i, (x_batch, y_batch) in enumerate(self.test_ds):
      y_hat = self.model(x_batch, training=False).numpy()
      predictions = np.append(predictions, y_hat.reshape(-1,))
      
    # reshape prediction shape = (n_samples * n_provinces)
    # to original test shape (n_samples, n_provinces)
    predictions = np.reshape(predictions, (self.raw_test.shape))
    self.predictions = predictions

    # Inverse predictions to normal range
    self.callbacks.on_predict_end(logs=None)
    return self.predictions