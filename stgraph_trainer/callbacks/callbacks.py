import tensorflow as tf
from ..utils import compose

class PostPredictionCallback(tf.keras.callbacks.Callback):
  """
  Callback to post process prediction.

  Parameters
  ----------
  funcs: list, object, or function
    Functions to apply on model's original prediction.
  """
  def __init__(self, funcs=None):
    super(PostPredictionCallback, self).__init__()
    self.funcs = funcs

  def on_test_batch_end(self, batch, logs=None):
    """
    Post-process prediction after test batch.
    Functions will be applied on `y_hat` which is the prediction for a batch.
    """
    if self.params['trainer'].y_hat is not None:
      y_hat = compose(self.params['trainer'].y_hat,
                      funcs=self.funcs,
                      idx=self.params['trainer'].idx)
      self.params['trainer'].y_hat = y_hat

  def on_predict_end(self, logs=None):
    """
    Post-process prediction on predict end.
    Functions will be applied on `predictions` which is the prediction for all dataset.
    """
    if self.params['trainer'].predictions is not None:
      predictions = compose(self.params['trainer'].predictions,
                            funcs=self.funcs)
      self.params['trainer'].predictions = predictions