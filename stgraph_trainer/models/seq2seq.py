import tensorflow as tf

class Encoder(tf.keras.Model):
  """
  Encoder to encode information from the input sequence.

  Parameters
  ----------
  input_size: int
    The number of input units to the GRU layer.

  enc_units: int
    The number of GRU's output units.

  batch_size: int
    The number of samples in a data batch.

  dropout: float, default: 0
    Fraction of the units to drop for the linear transformation.

  recurrent_dropout: float, default: 0
    Fraction of the units to drop for the linear transformation of the recurrent state.
  """
  def __init__(self,
               input_size,
               enc_units,
               batch_size,
               dropout=0.0,
               recurrent_dropout=0.0):
    super(Encoder, self).__init__()
    self.input_size = input_size
    self.enc_units = enc_units
    self.batch_size = batch_size
    self.fc = tf.keras.layers.Dense(self.input_size)
    self.gru = tf.keras.layers.GRU(self.enc_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform',
                                   dropout=dropout,
                                   recurrent_dropout=recurrent_dropout)
  
  def call(self, x, hidden, training=True):
    """
    Pass input through the model.

    Parameters
    ----------
    x: array shape of (batch_size, sequence_length, n_features)
      Input sequence data, consisting of sequence_length time steps and n_features features.

    hidden: array shape of (batch_size, enc_units)
      Initial state of the GRU, having the shape of batch_size x the number of GRU's output units.

    training: bool, default: True
      A boolean value indicating the training/testing state of the method call.

    Returns
    -------
    output: array shape of (batch_size, sequence_length, enc_units)
      The full sequence output of the GRU.

    state: array shape of (batch_size, enc_units)
      The final state of the GRU.
    """
    if len(x.shape) == 2:
      x = tf.reshape(x, (x.shape[0], x.shape[1], 1))
    x = self.fc(x, training=training)
    output, state = self.gru(x, initial_state=hidden, training=training)
    return output, state

  def initialize_hidden_state(self):
    """
    Initialize a zero hidden state for the encoder.

    Returns
    -------
    hidden_state: array shape of (batch_size, enc_units)
      A zero hidden state with the shape of batch_size x the number of GRU's output units.
    """
    return tf.zeros((self.batch_size, self.enc_units))

class Decoder(tf.keras.Model):
  """
  Decoder to decode information from the encoded sequence.

  Parameters
  ----------
  output_size: int
    The number of output units in the last dense layer.

  dec_units: int
    The number of the GRU's output units.

  batch_size: int
    The number of samples in a data batch.

  dropout: float, default: 0
    Fraction of the units to drop for the linear transformation.

  recurrent_dropout: float, default: 0
    Fraction of the units to drop for the linear transformation of the recurrent state.
  """
  def __init__(self,
               output_size,
               dec_units,
               batch_size,
               dropout=0.0,
               recurrent_dropout=0.0):
    super(Decoder, self).__init__()
    self.dec_units = dec_units
    self.batch_size = batch_size
    self.gru = tf.keras.layers.GRU(self.dec_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform',
                                   dropout=dropout,
                                   recurrent_dropout=recurrent_dropout)
    self.fc = tf.keras.layers.Dense(output_size)

  def call(self, x, hidden, enc_output=None, training=True):
    """
    Pass input through the model.

    Parameters
    ----------
    x: array shape of (batch_size, sequence_length, n_features)
      Input sequence data, consisting of sequence_length time steps and n_features features.

    hidden: array shape of (batch_size, dec_units)
      Hidden state of the GRU, having the shape of batch_size x the number of GRU's output units.

    enc_output: array shape of, default: None
      The full sequence output from the encoder. It is used for computing attention weights.
      So, there is nothing to do with it in this method now.

    training: bool, default: True
      A boolean value indicating the training/testing state of the method call.

    Returns
    -------
    x: array shape of (batch_size, output_size)
      The final decoded output (or prediction) of the decoder.

    hidden_state: array shape of (batch_size, dec_units)
      The final state of the GRU.
    """
    if len(x.shape) == 2:
      x = tf.reshape(x, (x.shape[0], x.shape[1], 1))
    decoding_output, hidden_state = self.gru(x,
                                             initial_state=hidden,
                                             training=training)
    decoding_output = tf.reshape(decoding_output, (-1, decoding_output.shape[2]))
    x = self.fc(decoding_output)
    return x, hidden_state

class BahdanauAttention(tf.keras.layers.Layer):
  """
  Bahdanau Attention mechanism.

  Parameters
  ----------
  units: int
    The number of units in the attention mechanism.
  """
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, query, values):
    """
    Compute attention weights for query and values.

    Parameters
    ----------
    query: array shape of (batch_size, hidden_size)
      The input to compute the attention weights against the `values`.

    values: array shape of (batch_size, sequence_len, hidden_size)
      The reference to compute the attention weights for the `query`.

    Returns
    -------
    context_vector: array shape of (batch_size, hidden_size)
      The vector contains weighted information (summation) from the `values` variables.

    attention_weights: array shape of (batch_size, sequence_length, 1)
      The attention weights (scores) of the `query` corresponding to each unit (in the sequence) of the `values`.
    """
    # query_with_time_axis shape = (batch_size, 1, hidden_size)
    # we are doing this to broadcast addition along the time axis to calculate the score
    query_with_time_axis = tf.expand_dims(query, 1)

    # score shape == (batch_size, sequence_len, 1)
    # we get 1 at the last axis because we are applying score to self.V
    # the shape of the tensor before applying self.V is (batch_size, max_length, units)
    score = self.V(tf.nn.tanh(self.W1(query_with_time_axis) + self.W2(values)))

    # attention_weights shape == (batch_size, max_length, 1)
    attention_weights = tf.nn.softmax(score, axis=1)

    # context_vector shape after sum == (batch_size, hidden_size)
    context_vector = attention_weights * values
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights

class DecoderWithAttention(tf.keras.Model):
  """
  Decoder with Attention mechanism to decode information from the encoded sequence.

  Parameters
  ----------
  output_size: int
    The number of output units in the last dense layer.

  dec_units: int
    The number of the GRU's output units.

  batch_size: int
    The number of samples in a data batch.

  dropout: float, default: 0
    Fraction of the units to drop for the linear transformation.

  recurrent_dropout: float, default: 0
    Fraction of the units to drop for the linear transformation of the recurrent state.
  """
  def __init__(self,
               output_size,
               dec_units,
               batch_size,
               dropout=0.0,
               recurrent_dropout=0.0):
    super(DecoderWithAttention, self).__init__()
    self.dec_units = dec_units
    self.batch_size = batch_size
    self.gru = tf.keras.layers.GRU(self.dec_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform',
                                   dropout=dropout,
                                   recurrent_dropout=recurrent_dropout)
    self.fc = tf.keras.layers.Dense(output_size)

    # used for attention
    self.attention = BahdanauAttention(self.dec_units)

  def call(self, x, hidden, enc_output=None, training=True):
    """
    Pass input through the model.

    Parameters
    ----------
    x: array shape of (batch_size, sequence_length, n_features)
      Input sequence data, consisting of sequence_length time steps and n_features features.

    hidden: array shape of (batch_size, dec_units)
      Hidden state of the GRU, having the shape of batch_size x the number of GRU's output units.

    enc_output: array shape of, default: None
      The full sequence output from the encoder. It is used for computing attention weights.

    training: bool, default: True
      A boolean value indicating the training/testing state of the method call.

    Returns
    -------
    x: array shape of (batch_size, output_size)
      The final decoded output (or prediction) of the decoder.

    hidden_state: array shape of (batch_size, dec_units)
      The final state of the GRU.
    """
    if len(x.shape) == 2:
      x = tf.reshape(x, (x.shape[0], x.shape[1], 1))
    
    context_vector, attention_weights = self.attention(hidden, enc_output)
    x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
    decoding_output, hidden_state = self.gru(x, training=training)
    decoding_output = tf.reshape(decoding_output, (-1, decoding_output.shape[2]))
    x = self.fc(decoding_output)
    return x, hidden_state