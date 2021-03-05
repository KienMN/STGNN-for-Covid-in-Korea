import tensorflow as tf
import numpy as np
import math
import time
from .base import BaseTrainer

class Seq2SeqTrainer(BaseTrainer):
  """
  Trainer class for the Seq2Seq model.

  Parameters
  ----------
  model: array of 2 objects
    The encoder and decoder.

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
    Here, n_features is the number of predicted time series.

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
    super(Seq2SeqTrainer, self).__init__(model,
                                         train_ds,
                                         test_ds,
                                         loss_func,
                                         optimizer,
                                         callbacks)
    if len(model) != 2:
      raise Exception('Model needs an encoder and a decoder')
    self.encoder, self.decoder = model
    
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
      enc_hidden = self.encoder.initialize_hidden_state()

      start_epoch_train_time = time.time()
      for x_batch, y_batch in self.train_ds:
        loss = self.train_step(x_batch, y_batch, enc_hidden)
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

  def train_step(self, x_batch, y_batch, enc_hidden):
    """
    Train the model for 1 step (or batch).

    Parameters
    ----------
    x_batch: array shape of (batch_size, time_steps)
      Input batch data for features.

    y_batch: array shape of (batch_size, n_timesteps_ahead)
      Input batch data for labels with the number of timesteps ahead to predict.

    enc_hidden: array shape of (batch_size, enc_hidden_size)
      Hidden state for the encoder.

    Returns
    -------
    loss: float Tensor
      Training loss of a batch.
    """
    loss = 0.
    with tf.GradientTape() as tape:
      enc_output, enc_hidden = self.encoder(x_batch, enc_hidden)
      dec_hidden = enc_hidden
      dec_input = tf.expand_dims(x_batch[:, -1], 1)

      for t in range(0, y_batch.shape[1]):
        dec_output, dec_hidden = self.decoder(dec_input, dec_hidden, enc_output)
        loss += self.loss_func(tf.expand_dims(y_batch[:, t], 1), dec_output)
        # Using teacher forcing
        dec_input = tf.expand_dims(y_batch[:, t], 1)

    batch_loss = loss / y_batch.shape[1]
    variables = self.encoder.trainable_variables + self.decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    self.optimizer.apply_gradients(zip(gradients, variables))
    return batch_loss

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
    predictions = np.array([])
    hidden = tf.zeros((1, self.encoder.enc_units))
    for i, (x_batch, y_batch) in enumerate(self.test_ds):
      y_pred = []
      hidden = tf.zeros((x_batch.shape[0], self.encoder.enc_units))

      enc_output, enc_hidden = self.encoder(x_batch, hidden, training=False)
      dec_hidden = enc_hidden
      dec_input = tf.expand_dims(x_batch[:, -1], 1)

      for t in range(0, y_batch.shape[1]):
        dec_output, dec_hidden = self.decoder(dec_input,
                                              dec_hidden,
                                              enc_output,
                                              training=False)
        
        pred = dec_output.numpy()
        y_pred.append(pred)
        dec_input = tf.expand_dims(dec_output[:, -1], 1)
      y_pred = np.concatenate(y_pred, axis=-1)
      predictions = np.append(predictions, y_pred.reshape(-1,))
    
    # reshape prediction shape = (n_samples * n_provinces)
    # to original test shape (n_samples, n_provinces)
    predictions = np.reshape(predictions, (self.raw_test.shape))
    self.predictions = predictions

    # Inverse predictions to normal range
    self.callbacks.on_predict_end(logs=None)
    return self.predictions