import tensorflow as tf
from ..utils import compose

class PostPredictionCallback(tf.keras.callbacks.Callback):
  def __init__(self, funcs=None):
    super(PostPredictionCallback, self).__init__()
    self.funcs = funcs

  def on_test_batch_end(self, batch, logs=None):
    if self.params['trainer'].y_hat is not None:
      y_hat = compose(self.params['trainer'].y_hat,
                      funcs=self.funcs,
                      idx=self.params['trainer'].idx)
      self.params['trainer'].y_hat = y_hat

  def on_predict_end(self, logs=None):
    if self.params['trainer'].predictions is not None:
      predictions = compose(self.params['trainer'].predictions,
                            funcs=self.funcs)
      self.params['trainer'].predictions = predictions