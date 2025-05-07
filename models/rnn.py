import tensorflow as tf

path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
# The unique characters in the file
vocab = sorted(set(text))
# Length of the vocabulary in StringLookup Layer
ids_from_chars = tf.keras.layers.StringLookup(
    vocabulary=list(vocab), mask_token=None)
vocab_size_rnn = len(ids_from_chars.get_vocabulary())
# The embedding dimension
embedding_dim = 256
# Number of RNN units
rnn_units = 1024
seq_length = 100
ss_batch_size=8
ss_buffer_size=100


def create_rnn_model(vocab_size, embedding_dim, rnn_units):
    inputs = tf.keras.Input(shape=(None,))
    x = tf.keras.layers.Embedding(vocab_size, embedding_dim)(inputs)
    x, states = tf.keras.layers.GRU(rnn_units, return_sequences=True, return_state=True, activation='tanh', recurrent_activation='sigmoid', recurrent_dropout=0, unroll=False, use_bias=True, reset_after=True)(x)
    outputs = tf.keras.layers.Dense(vocab_size)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

class FlattenedCategoricalAccuracy(tf.keras.metrics.SparseCategoricalAccuracy):

  def __init__(self, name='sparse_categorical_accuracy', dtype=tf.float32):
    super().__init__(name, dtype=dtype)

  def update_state(self, y_true, y_pred, sample_weight=None):
    y_true = tf.reshape(y_true, [-1, 1])
    y_pred = tf.reshape(y_pred, [-1, vocab_size_rnn, 1])
    return super().update_state(y_true, y_pred, sample_weight)


def evaluate_rnn_model(keras_model, dataset):
  # Take our global model weights and push them back into a Keras model to
  # use its standard `.evaluate()` method.
  keras_model.compile(
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=[FlattenedCategoricalAccuracy()])
  loss, accuracy = keras_model.evaluate(dataset, steps=2, verbose=0)
  #print('\tEval: loss={l:.3f}, accuracy={a:.3f}'.format(l=loss, a=accuracy))
  return loss, accuracy
