import tensorflow as tf
import tensorflow_federated as tff
import collections
from typing import Callable, Tuple


path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
# The unique characters in the file
vocab = sorted(set(text))

# Construct a lookup table to map string chars to indexes,
# using the vocab loaded above:
table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(
        keys=vocab, values=tf.constant(list(range(len(vocab))),
                                       dtype=tf.int64)),
    default_value=0)


def to_ids(x):
  s = tf.reshape(x['snippets'], shape=[1])
  chars = tf.strings.bytes_split(s).values
  ids = table.lookup(chars)
  return ids


def split_input_target(chunk):
  input_text = tf.map_fn(lambda x: x[:-1], chunk)
  target_text = tf.map_fn(lambda x: x[1:], chunk)
  return (input_text, target_text)

def create_preprocess_train_dataset(buffer_size, client_epochs_per_round, batch_size, drop_remainder, seq_length) -> Callable[[tf.data.Dataset], tf.data.Dataset]:
    def preprocess_train_dataset(dataset):
      return (
          # Map ASCII chars to int64 indexes using the vocab
          dataset.map(to_ids)
          # Split into individual chars
          .unbatch()
          # Form example sequences of SEQ_LENGTH +1
          .batch(seq_length + 1, drop_remainder=drop_remainder)
          # Shuffle and form minibatches
          .shuffle(buffer_size).batch(batch_size, drop_remainder=drop_remainder)
          .repeat(count=client_epochs_per_round)
          # And finally split into (input, target) tuples,
          # each of length SEQ_LENGTH.
          .map(split_input_target))
    return preprocess_train_dataset

def create_preprocess_test_dataset(buffer_size, test_batch_size, drop_remainder, seq_length) -> Callable[[tf.data.Dataset], tf.data.Dataset]:
    def preprocess_test_dataset(dataset):
        return (
          # Map ASCII chars to int64 indexes using the vocab
          dataset.map(to_ids)
          # Split into individual chars
          .unbatch()
          # Form example sequences of SEQ_LENGTH +1
          .batch(seq_length + 1, drop_remainder=drop_remainder)
          .shuffle(buffer_size).batch(test_batch_size, drop_remainder=drop_remainder)
          # And finally split into (input, target) tuples,
          # each of length SEQ_LENGTH.
          .map(split_input_target))
    return preprocess_test_dataset
        

def get_federated_shakespeare_datasets(
    batch_size=8, # 'Batch size used on the client
    test_batch_size=8, #Minibatch size of test data
    client_epochs_per_round=1,
    buffer_size=100,
    drop_remainder=False,
    seq_length=100
) -> Tuple[tff.simulation.datasets.ClientData,
           tff.simulation.datasets.ClientData]:
    shakespeare_train, shakespeare_test = tff.simulation.datasets.shakespeare.load_data()
    client_datasets = list(shakespeare_train.client_ids)

    preprocess_train_fn = create_preprocess_train_dataset(
        buffer_size = buffer_size,
        client_epochs_per_round = client_epochs_per_round,
        batch_size = batch_size,
        drop_remainder = drop_remainder,
        seq_length = seq_length
    )
    preprocess_test_fn = create_preprocess_test_dataset(
        buffer_size = buffer_size,
        test_batch_size = test_batch_size,
        drop_remainder = drop_remainder,
        seq_length = seq_length
    )

    federated_train_data = shakespeare_train.preprocess(preprocess_train_fn)
    #federated_test_data = shakespeare_test.preprocess(preprocess_test_fn)
    
    return federated_train_data

def get_centralized_shakespeare_datasets(
    batch_size=8, # 'Batch size used on the client
    test_batch_size=8, #Minibatch size of test data
    client_epochs_per_round=1,
    buffer_size=100,
    drop_remainder=False,
    seq_length=100
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:

    shakespeare_train, shakespeare_test = tff.simulation.datasets.shakespeare.load_data()
    # Split the client datasets into training and validation datasets
    # Get the list of client IDs
    client_ids = shakespeare_train.client_ids
    
    preprocess_train_fn = create_preprocess_train_dataset(
        buffer_size = buffer_size,
        client_epochs_per_round = client_epochs_per_round,
        batch_size = batch_size,
        drop_remainder = drop_remainder,
        seq_length = seq_length
    )
    preprocess_test_fn = create_preprocess_test_dataset(
        buffer_size = buffer_size,
        test_batch_size = test_batch_size,
        drop_remainder = drop_remainder,
        seq_length = seq_length
    )

    centralized_train_data = preprocess_train_fn(shakespeare_train.create_tf_dataset_from_all_clients())
    centralized_test_data = preprocess_test_fn(shakespeare_test.create_tf_dataset_from_all_clients())
    centralized_validation_data = preprocess_test_fn(shakespeare_test.create_tf_dataset_from_all_clients())
    
    return centralized_train_data, centralized_validation_data, centralized_test_data