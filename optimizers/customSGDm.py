import collections
from typing import Any, Optional, TypeVar
import tensorflow as tf
from tensorflow_federated.python.common_libs import structure
from tensorflow_federated.python.learning.optimizers import optimizer

_MOMENTUM_KEY = 'momentum'
_ACCUMULATOR_KEY = 'accumulator'
_CLIPNORM_KEY = 'clipnorm'

State = TypeVar('State', bound=collections.OrderedDict[str, Any])
Hparams = TypeVar('Hparams', bound=collections.OrderedDict[str, float])


class _SGD(optimizer.Optimizer[State, optimizer.Weights, Hparams]):
    """Gradient descent optimizer with momentum and clipnorm support."""

    def __init__(
            self,
            learning_rate: float,
            momentum: Optional[float] = None,
            clipnorm: Optional[float] = None,
    ):
        """Initializes SGD optimizer."""
        if learning_rate < 0.0:
            raise ValueError(
                f'SGD `learning_rate` must be nonnegative, found {learning_rate}.'
            )
        if momentum:
            if momentum < 0.0 or momentum > 1.0:
                raise ValueError(
                    'SGD `momentum` must be `None` or in the range [0, 1], found '
                    f'{momentum}.'
                )
        self._hparams_keys = [optimizer.LEARNING_RATE_KEY, _CLIPNORM_KEY]
        if momentum:
            self._hparams_keys.append(_MOMENTUM_KEY)
        self._lr = learning_rate
        self._momentum = momentum
        self._clipnorm = clipnorm

    def initialize(self, specs: Any) -> State:
        state = collections.OrderedDict([(optimizer.LEARNING_RATE_KEY, self._lr), (_CLIPNORM_KEY, self._clipnorm)])
        if self._momentum is not None and self._momentum > 0:
            state[_MOMENTUM_KEY] = self._momentum
            state[_ACCUMULATOR_KEY] = tf.nest.map_structure(
                lambda s: tf.zeros(s.shape, s.dtype), specs
            )
        return state

    def next(
            self, state: State, weights: optimizer.Weights, gradients: Any
    ) -> tuple[State, optimizer.Weights]:
        gradients = optimizer.handle_indexed_slices_gradients(gradients)
        optimizer.check_weights_gradients_match(weights, gradients)
        lr = state[optimizer.LEARNING_RATE_KEY]
        if _MOMENTUM_KEY not in state:
            updated_weights = tf.nest.map_structure(
                lambda w, g: w - lr * g, weights, gradients
            )
            updated_state = collections.OrderedDict(
                [(optimizer.LEARNING_RATE_KEY, lr), (_CLIPNORM_KEY, self._clipnorm)]
            )
        else:
            momentum = state[_MOMENTUM_KEY]
            accumulator = state[_ACCUMULATOR_KEY]
            optimizer.check_weights_state_match(weights, accumulator, 'accumulator')
            updated_accumulator = tf.nest.map_structure(
                lambda a, g: momentum * a + g, accumulator, gradients
            )
            if self._clipnorm:
                updated_accumulator = tf.clip_by_global_norm(updated_accumulator, self._clipnorm)[0]
            updated_weights = tf.nest.map_structure(
                lambda w, m: w - lr * m, weights, updated_accumulator
            )
            updated_state = collections.OrderedDict([
                (optimizer.LEARNING_RATE_KEY, lr),
                (_MOMENTUM_KEY, momentum),
                (_ACCUMULATOR_KEY, updated_accumulator),
                (_CLIPNORM_KEY, self._clipnorm)
            ])
        return updated_state, updated_weights

    def get_hparams(self, state: State) -> Hparams:
        return collections.OrderedDict([(k, state[k]) for k in self._hparams_keys])

    def set_hparams(self, state: State, hparams: Hparams) -> State:
        return structure.update_struct(state, **hparams)



def build_custom_sgdm(
        learning_rate: float = 0.01,
        momentum: Optional[float] = None,
        clipnorm: Optional[float] = None
) -> optimizer.Optimizer:
    return _SGD(learning_rate=learning_rate, momentum=momentum, clipnorm=clipnorm)
