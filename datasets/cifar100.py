import tensorflow as tf
import tensorflow_federated as tff
import collections
from typing import Callable, Tuple

def cifar100_element_fn(element):
    """
    Preprocess a single element from the CIFAR-100 dataset.

    Normalizes the image to [0, 1] and reshapes the label to a scalar.
    """
    image = tf.cast(element['image'], tf.float32) / 255.0
    # Use tf.reshape to ensure label is a scalar regardless of its input shape.
    label = tf.reshape(element['label'], [])
    return collections.OrderedDict(x=image, y=label)


def create_preprocess_train_dataset_cifar100(buffer_size: int,
                                             client_epochs_per_round: int,
                                             batch_size: int,
                                             drop_remainder: bool) -> Callable[[tf.data.Dataset], tf.data.Dataset]:
    """
    Returns a function that preprocesses a training dataset for CIFAR-100.
    
    This function applies:
      - mapping using cifar100_element_fn,
      - shuffling with a provided buffer size,
      - repeating for the specified number of epochs,
      - batching with the specified batch size.
    """
    def preprocess_train_dataset(dataset):
        return (
            dataset.map(cifar100_element_fn)
                   .shuffle(buffer_size)
                   .repeat(client_epochs_per_round)
                   .batch(batch_size, drop_remainder)
        )
    return preprocess_train_dataset

def create_preprocess_test_dataset_cifar100(test_batch_size: int,
                                            drop_remainder: bool) -> Callable[[tf.data.Dataset], tf.data.Dataset]:
    """
    Returns a function that preprocesses a test dataset for CIFAR-100.
    
    This function applies mapping using cifar100_element_fn and batches the dataset.
    """
    def preprocess_test_dataset(dataset):
        return dataset.map(cifar100_element_fn).batch(test_batch_size, drop_remainder)
    return preprocess_test_dataset

def get_federated_cifar100_datasets(
    batch_size: int = 16,
    test_batch_size: int = 128,
    train_clients_per_round: int = 2,
    client_epochs_per_round: int = 1,
    buffer_size: int = 500,  # Adjust based on the expected client dataset size.
    drop_remainder: bool = False
) -> Tuple[tff.simulation.datasets.ClientData, tff.simulation.datasets.ClientData]:
    """
    Loads and preprocesses the federated CIFAR-100 dataset.
    
    The training split has 500 clients (client IDs "0" to "499") and the test split 100 clients.
    """
    cifar100_train, cifar100_test = tff.simulation.datasets.cifar100.load_data()
    
    client_ids = cifar100_train.client_ids
   
    preprocess_train_fn = create_preprocess_train_dataset_cifar100(
        buffer_size=buffer_size,
        client_epochs_per_round=client_epochs_per_round,
        batch_size=batch_size,
        drop_remainder=drop_remainder
    )
    preprocess_test_fn = create_preprocess_test_dataset_cifar100(
        test_batch_size=test_batch_size,
        drop_remainder=drop_remainder
    )

    federated_train_data = cifar100_train.preprocess(preprocess_train_fn)
    federated_test_data = cifar100_test.preprocess(preprocess_test_fn)
    
    return federated_train_data, federated_test_data

def get_centralized_cifar100_datasets(
    batch_size: int = 16,
    test_batch_size: int = 128,
    train_clients_per_round = 2,
    client_epochs_per_round: int = 1,
    buffer_size: int = 500,
    drop_remainder: bool = False
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """
    Loads and preprocesses centralized CIFAR-100 datasets for training, validation, and testing.
    
    Note: The test clients form a true partition of the CIFAR-100 testing split.
    """
    cifar100_train, cifar100_test = tff.simulation.datasets.cifar100.load_data()
    
    client_ids = cifar100_train.client_ids
   
    preprocess_train_fn = create_preprocess_train_dataset_cifar100(
        buffer_size=buffer_size,
        client_epochs_per_round=client_epochs_per_round,
        batch_size=batch_size,
        drop_remainder=drop_remainder
    )
    preprocess_test_fn = create_preprocess_test_dataset_cifar100(
        test_batch_size=test_batch_size,
        drop_remainder=drop_remainder
    )

    centralized_train_data = preprocess_train_fn(
        cifar100_train.create_tf_dataset_from_all_clients())
    centralized_test_data = preprocess_test_fn(
        cifar100_test.create_tf_dataset_from_all_clients())
    centralized_validation_data = preprocess_test_fn(
        cifar100_test.create_tf_dataset_from_all_clients())

    return centralized_train_data, centralized_validation_data, centralized_test_data
