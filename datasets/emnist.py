import tensorflow as tf
import tensorflow_federated as tff
import collections
from typing import Callable, Tuple


def element_fn(element):
    return collections.OrderedDict(
        x=tf.expand_dims(element['pixels'], -1), y=element['label']
    )

def create_preprocess_train_dataset(buffer_size, client_epochs_per_round, batch_size, drop_remainder) -> Callable[[tf.data.Dataset], tf.data.Dataset]:
    def preprocess_train_dataset(dataset):
        # Use buffer_size same as the maximum client dataset size,
        # 418 for Federated EMNIST
        return (
            dataset.map(element_fn)
            .shuffle(buffer_size)
            .repeat(count=client_epochs_per_round)
            .batch(batch_size, drop_remainder)
        )
    return preprocess_train_dataset

def create_preprocess_test_dataset(test_batch_size, drop_remainder) -> Callable[[tf.data.Dataset], tf.data.Dataset]:
    def preprocess_test_dataset(dataset):
        return dataset.map(element_fn).batch(
            test_batch_size, drop_remainder
        )
    return preprocess_test_dataset
        

def get_federated_emnist_datasets(
    batch_size=16, # 'Batch size used on the client
    test_batch_size=128, #Minibatch size of test data
    train_clients_per_round=2,
    client_epochs_per_round=1,
    buffer_size=418,
    drop_remainder=False
) -> Tuple[tff.simulation.datasets.ClientData,
           tff.simulation.datasets.ClientData]:
    emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data(only_digits=False)
    client_datasets = list(emnist_train.client_ids)

    # Split the client datasets into training and validation datasets
    # Get the list of client IDs
    client_ids = emnist_train.client_ids
    
    preprocess_train_fn = create_preprocess_train_dataset(
        buffer_size = buffer_size,
        client_epochs_per_round = client_epochs_per_round,
        batch_size = batch_size,
        drop_remainder = drop_remainder
    )
    preprocess_test_fn = create_preprocess_test_dataset(
        test_batch_size = test_batch_size,
        drop_remainder = drop_remainder
    )

    federated_train_data = emnist_train.preprocess(preprocess_train_fn)
    federated_test_data = emnist_test.preprocess(preprocess_test_fn)
    
    return federated_train_data, federated_test_data

def get_centralized_emnist_datasets(
    batch_size=16, # 'Batch size used on the client
    test_batch_size=128, #Minibatch size of test data
    train_clients_per_round=2,
    client_epochs_per_round=1,
    buffer_size=418,
    drop_remainder=False
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:

    emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data(only_digits=False)
    # Split the client datasets into training and validation datasets
    # Get the list of client IDs
    client_ids = emnist_train.client_ids

    preprocess_train_fn = create_preprocess_train_dataset(
        buffer_size = buffer_size,
        client_epochs_per_round = client_epochs_per_round,
        batch_size = batch_size,
        drop_remainder = drop_remainder
    )
    preprocess_test_fn = create_preprocess_test_dataset(
        test_batch_size = test_batch_size,
        drop_remainder = drop_remainder
    )

    centralized_train_data = preprocess_train_fn(emnist_train.create_tf_dataset_from_all_clients())
    centralized_test_data = preprocess_test_fn(emnist_test.create_tf_dataset_from_all_clients())
    centralized_validation_data = preprocess_test_fn(emnist_test.create_tf_dataset_from_all_clients())
    
    return centralized_train_data, centralized_validation_data, centralized_test_data