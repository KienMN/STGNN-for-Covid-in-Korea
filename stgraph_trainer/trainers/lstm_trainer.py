import tensorflow as tf
import numpy as np
from .base import BaseTrainer
import math
import time

class LSTMTrainer(BaseTrainer):
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
    with tf.GradientTape() as tape:
      y_pred = self.model(x_batch, training=True)
      assert y_pred.shape == y_batch.shape
      loss = self.loss_func(y_true=y_batch, y_pred=y_pred)
    gradients = tape.gradient(loss, self.model.trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
    return loss

  def evaluate(self):
    predictions = self.predict()
    return self.loss_func(self.raw_test, predictions)

  def predict(self):
    predictions = []
    for i, (x_batch, y_batch) in enumerate(self.test_ds):
      self.y_hat = self.model(x_batch, training=False)
      self.idx = i

      # Place for callback to inverse results
      self.callbacks.on_test_batch_end(batch=i)
      predictions.append(self.y_hat.copy())
    return np.concatenate(predictions, axis=0)