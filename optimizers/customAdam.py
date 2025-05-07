import collections
from typing import Any, TypeVar
import tensorflow as tf
from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.learning.optimizers import optimizer

_BETA_1_KEY = 'beta_1'
_BETA_2_KEY = 'beta_2'
_EPSILON_KEY = 'epsilon'
_STEP_KEY = 'step'
_ACCUMULATOR_KEY = 'accumulator'
_PRECONDITIONER_KEY = 'preconditioner'
_CLIPNORM_KEY = 'clipnorm'
_HPARAMS_KEYS = [
    optimizer.LEARNING_RATE_KEY,
    _BETA_1_KEY,
    _BETA_2_KEY,
    _EPSILON_KEY,
    _CLIPNORM_KEY,
]

State = TypeVar('State', bound=collections.OrderedDict[str, Any])
Hparams = TypeVar('Hparams', bound=collections.OrderedDict[str, float])

class _Adam(optimizer.Optimizer[State, optimizer.Weights, Hparams]):
    """Adam optimizer, see `build_adam` for details."""

    def __init__(
        self,
        learning_rate: float,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-7,
        clipnorm: float = None,
    ):
        """Initializes Adam optimizer."""
        if learning_rate < 0.0:
            raise ValueError(
                f'Adam `learning_rate` must be nonnegative, found {learning_rate}.'
            )
        if beta_1 < 0.0 or beta_1 > 1.0:
            raise ValueError(
                f'Adam `beta_1` must be in the range [0.0, 1.0], found {beta_1}.'
            )
        if beta_2 < 0.0 or beta_2 > 1.0:
            raise ValueError(
                f'Adam `beta_2` must be in the range [0.0, 1.0], found {beta_2}.'
            )
        if epsilon < 0.0:
            raise ValueError(f'Adam `epsilon` must be nonnegative, found {epsilon}.')
        self._lr = learning_rate
        self._beta_1 = beta_1
        self._beta_2 = beta_2
        self._epsilon = epsilon
        self._clipnorm = clipnorm
        

    def initialize(self, specs: Any) -> State:
        """Initializes the optimizer state."""
        initial_accumulator = tf.nest.map_structure(
            lambda s: tf.zeros(s.shape, s.dtype), specs
        )
        initial_preconditioner = tf.nest.map_structure(
            lambda s: tf.zeros(s.shape, s.dtype), specs
        )
        state = collections.OrderedDict([
            (optimizer.LEARNING_RATE_KEY, self._lr),
            (_BETA_1_KEY, self._beta_1),
            (_BETA_2_KEY, self._beta_2),
            (_EPSILON_KEY, self._epsilon),
            (_CLIPNORM_KEY, self._clipnorm),
            (_STEP_KEY, 0),
            (_ACCUMULATOR_KEY, initial_accumulator),
            (_PRECONDITIONER_KEY, initial_preconditioner),
        ])
        return state

    def next(
        self, state: State, weights: optimizer.Weights, gradients: Any
    ) -> tuple[State, optimizer.Weights]:
        gradients = optimizer.handle_indexed_slices_gradients(gradients)
        optimizer.check_weights_gradients_match(weights, gradients)
        lr = state[optimizer.LEARNING_RATE_KEY]
        beta_1 = state[_BETA_1_KEY]
        beta_2 = state[_BETA_2_KEY]
        epsilon = state[_EPSILON_KEY]
        clipnorm = state[_CLIPNORM_KEY]
        step = state[_STEP_KEY] + 1
        accumulator = state[_ACCUMULATOR_KEY]
        preconditioner = state[_PRECONDITIONER_KEY]
        optimizer.check_weights_state_match(weights, accumulator, 'accumulator')
        optimizer.check_weights_state_match(weights, preconditioner, 'preconditioner')

        if self._clipnorm is not None:
            gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=clipnorm)
            
        updated_accumulator = tf.nest.map_structure(
            lambda a, g: a + (g - a) * (1 - beta_1), accumulator, gradients
        )
        updated_preconditioner = tf.nest.map_structure(
            lambda s, g: s + (tf.math.square(g) - s) * (1 - beta_2),
            preconditioner,
            gradients,
        )
        normalized_lr = (
            lr
            * tf.math.sqrt((1 - tf.math.pow(beta_2, tf.cast(step, tf.float32))))
            / (1 - tf.math.pow(beta_1, tf.cast(step, tf.float32)))
        )
        
        updated_weights = tf.nest.map_structure(
            lambda w, g, a, s: w - normalized_lr * a / (tf.math.sqrt(s) + epsilon),
            weights,
            gradients,
            updated_accumulator,
            updated_preconditioner,
        )

        updated_state = collections.OrderedDict(
            [
                (optimizer.LEARNING_RATE_KEY, lr),
                (_BETA_1_KEY, beta_1),
                (_BETA_2_KEY, beta_2),
                (_EPSILON_KEY, epsilon),
                (_STEP_KEY, step),
                (_ACCUMULATOR_KEY, updated_accumulator),
                (_PRECONDITIONER_KEY, updated_preconditioner),
                (_CLIPNORM_KEY, clipnorm),
            ]
        )
        return updated_state, updated_weights


def build_custom_adam(
    learning_rate: float,
    beta_1: float = 0.9,
    beta_2: float = 0.999,
    epsilon: float = 1e-7,
    clipnorm: float = None,
) -> optimizer.Optimizer:
    return _Adam(learning_rate, beta_1, beta_2, epsilon, clipnorm)
