import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

def create_cifar100_cnn_model():
    """
    Creates a CNN model for CIFAR-100.

    The model is designed for 32×32 RGB images and outputs logits for 100 classes.
    The architecture includes two convolutional blocks with dropout and a dense classification head.
    
    Returns:
      An uncompiled tf.keras.Model.
    """
    model = Sequential([
        # First convolutional block.
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        # Second convolutional block.
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        # Fully-connected layers.
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(100)  # 100 output logits for CIFAR-100 classes.
    ])
    return model

def evaluate_cifar100_cnn_model(keras_model, test_dataset):
    """
    Evaluates the accuracy of a Keras model on the CIFAR-100 test dataset.

    Args:
      keras_model: A Keras model that outputs logits.
      test_dataset: A tf.data.Dataset yielding batches in the format {'x': ..., 'y': ...}.
    
    Returns:
      The computed SparseCategoricalAccuracy metric value.
    """
    metric = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
    for batch in test_dataset:
        predictions = keras_model(batch['x'])
        metric.update_state(y_true=batch['y'], y_pred=predictions)
    return metric.result()
