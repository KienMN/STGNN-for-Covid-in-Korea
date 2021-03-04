from keras.callbacks import CallbackList

class BaseTrainer():
  """
  The Trainer base class containing all information for training and predicting.

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

  Attributes
  ----------
  history: dict of list
    Information of training. None for now.
  """
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
    """
    Train the model for 1 step (or batch).
    """
    pass

  def train(self, *args, **kwargs):
    """
    Train the model.
    """
    pass

  def evaluate(self, *args, **kwargs):
    """
    Conduct prediction and evaluation.
    """
    pass

  def predict(self, *args, **kwargs):
    """
    Make prediction.
    """
    pass