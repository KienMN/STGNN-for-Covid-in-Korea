import tensorflow as tf
from keras.callbacks import CallbackList

class BaseTrainer():
  def __init__(self,
               model,
               train_ds,
               test_ds,
               loss_func,
               optimizer,
               callbacks=None,
               *args,
               **kwargs):
    self.model = model
    self.train_ds = train_ds
    self.test_ds = test_ds
    self.loss_func = loss_func
    self.optimizer = optimizer
    if not isinstance(callbacks, CallbackList):
      callbacks = CallbackList(callbacks)
    self.callbacks = callbacks
    self.callbacks.set_params({'trainer': self})
    self.history = None

  def train_step(self, *args, **kwargs):
    pass

  def test_step(self, *args, **kwargs):
    pass

  def train(self, *args, **kwargs):
    pass

  def evaluate(self, *args, **kwargs):
    pass

  def predict(self, *args, **kwargs):
    pass