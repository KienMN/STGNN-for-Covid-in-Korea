import tensorflow as tf

class Encoder(tf.keras.Model):
  def __init__(self, input_size, enc_units, batch_size, dropout=0.0, recurrent_dropout=0.0):
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
    if len(x.shape) == 2:
      x = tf.reshape(x, (x.shape[0], x.shape[1], 1))
    x = self.fc(x, training=training)
    output, state = self.gru(x, initial_state=hidden, training=training)
    return output, state

  def initialize_hidden_state(self):
    return tf.zeros((self.batch_size, self.enc_units))

class Decoder(tf.keras.Model):
  def __init__(self, out_size, dec_units, batch_size, dropout=0.0, recurrent_dropout=0.0):
    super(Decoder, self).__init__()
    self.dec_units = dec_units
    self.batch_size = batch_size
    self.gru = tf.keras.layers.GRU(self.dec_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform',
                                   dropout=dropout,
                                   recurrent_dropout=recurrent_dropout)
    self.fc = tf.keras.layers.Dense(out_size)

  def call(self, x, hidden, enc_output=None, training=True):
    if len(x.shape) == 2:
      x = tf.reshape(x, (x.shape[0], x.shape[1], 1))
    decoding_output, hidden_state = self.gru(x, initial_state=hidden, training=training)
    decoding_output = tf.reshape(decoding_output, (-1, decoding_output.shape[2]))
    x = self.fc(decoding_output)
    return x, hidden_state

class BahdanauAttention(tf.keras.layers.Layer):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, query, values):
    # query hidden state shape == (batch_size, hidden_size)
    # query_with_time_axis shape = (batch_size, 1, hidden_size)
    # values shape == (batch_size, max_len, hidden_size)
    # we are doing this to broadcast addition along the time axis to calculate the score
    query_with_time_axis = tf.expand_dims(query, 1)

    # score shape == (batch_size, max_length, 1)
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
  def __init__(self, out_size, dec_units, batch_size, dropout=0.0, recurrent_dropout=0.0):
    super(DecoderWithAttention, self).__init__()
    self.dec_units = dec_units
    self.batch_size = batch_size
    self.gru = tf.keras.layers.GRU(self.dec_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform',
                                   dropout=dropout,
                                   recurrent_dropout=recurrent_dropout)
    self.fc = tf.keras.layers.Dense(out_size)

    # used for attention
    self.attention = BahdanauAttention(self.dec_units)

  def call(self, x, hidden, enc_output=None, training=True):
    if len(x.shape) == 2:
      x = tf.reshape(x, (x.shape[0], x.shape[1], 1))
    
    context_vector, attention_weights = self.attention(hidden, enc_output)
    x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
    decoding_output, hidden_state = self.gru(x, training=training)
    decoding_output = tf.reshape(decoding_output, (-1, decoding_output.shape[2]))
    x = self.fc(decoding_output)
    return x, hidden_state