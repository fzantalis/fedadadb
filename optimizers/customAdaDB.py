# Copyright 2021, The TensorFlow Federated Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""AdaDB optimizer."""

import collections
from typing import Any, TypeVar

import tensorflow as tf

from tensorflow_federated.python.common_libs import structure
from optimizers import nest_utils
from tensorflow_federated.python.learning.optimizers import optimizer

_BETA_1_KEY = 'beta_1'
_BETA_2_KEY = 'beta_2'
_EPSILON_KEY = 'epsilon'
_STEP_KEY = 'step'
_ACCUMULATOR_KEY = 'accumulator'
_PRECONDITIONER_KEY = 'preconditioner'
_ALPHA_STAR_KEY = 'alpha_star'
_CLIPNORM_KEY = 'clipnorm'

_HPARAMS_KEYS = [
    optimizer.LEARNING_RATE_KEY,
    _BETA_1_KEY,
    _BETA_2_KEY,
    _EPSILON_KEY,
    _ALPHA_STAR_KEY,
    _CLIPNORM_KEY,
]

State = TypeVar('State', bound=collections.OrderedDict[str, Any])
Hparams = TypeVar('Hparams', bound=collections.OrderedDict[str, Any])


class _AdaDB(optimizer.Optimizer[State, optimizer.Weights, Hparams]):
  """AdaDB optimizer, a data-bounded variant of Adam."""

  def __init__(
      self,
      learning_rate: float,
      beta_1 = 0.9,
      beta_2 = 0.999,
      epsilon = 1e-7,   
      alpha_star = 0.1,
      clipnorm = None,
  ):
    """Initializes the AdaDB optimizer."""
    if learning_rate < 0.0:
      raise ValueError(
          f'AdaDB `learning_rate` must be nonnegative, found {learning_rate}.'
      )
    if (beta_1 < 0.0 or beta_1 > 1.0):
      raise ValueError(
          f'AdaDB `beta_1` must be in the range [0.0, 1.0], found {beta_1}.'
      )
    if (beta_2 < 0.0 or beta_2 > 1.0):
      raise ValueError(
          f'AdaDB `beta_2` must be in the range [0.0, 1.0], found {beta_2}.'
      )
    if epsilon < 0.0:
      raise ValueError(
          f'AdaDB `epsilon` must be nonnegative, found {epsilon}.'
      )
    if alpha_star < 0.0:
      raise ValueError(
          f'AdaDB `alpha_star` must be nonnegative, found {alpha_star}.'
      )

    self._lr = learning_rate
    self._beta_1 = beta_1
    self._beta_2 = beta_2
    self._epsilon = epsilon
    self._alpha_star = alpha_star
    self._clipnorm = clipnorm

  def initialize(self, specs: Any) -> State:
    initial_accumulator = tf.nest.map_structure(
        lambda s: tf.zeros(s.shape, s.dtype), specs
    )
    initial_preconditioner = tf.nest.map_structure(
        lambda s: tf.zeros(s.shape, s.dtype), specs
    )
    return collections.OrderedDict([
        (optimizer.LEARNING_RATE_KEY, self._lr),
        (_BETA_1_KEY, self._beta_1),
        (_BETA_2_KEY, self._beta_2),
        (_EPSILON_KEY, self._epsilon),
        (_CLIPNORM_KEY, self._clipnorm),
        (_ALPHA_STAR_KEY, self._alpha_star),
        (_STEP_KEY, 0),
        (_ACCUMULATOR_KEY, initial_accumulator),
        (_PRECONDITIONER_KEY, initial_preconditioner),
    ])

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
    alpha_star = state[_ALPHA_STAR_KEY]
    step = state[_STEP_KEY] + 1
    accumulator = state[_ACCUMULATOR_KEY]
    preconditioner = state[_PRECONDITIONER_KEY]
    optimizer.check_weights_state_match(weights, accumulator, 'accumulator')
    optimizer.check_weights_state_match(
        weights, preconditioner, 'preconditioner'
    )

    if clipnorm is not None:
      gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=clipnorm)

    if tf.is_tensor(beta_1):
      casted_step = tf.cast(step, beta_1.dtype)
    else:
      casted_step = step

    normalized_lr = (
        lr * tf.math.sqrt(1.0 - tf.math.pow(beta_2, casted_step))
        / (1.0 - tf.math.pow(beta_1, casted_step))
    )

    def _adadb_update(w, a, p, g):
      if g is None:
        return w, a, p
      # Update biased first moment
      a = a + (g - a) * (1.0 - tf.cast(beta_1, a.dtype))
      # Update biased second moment
      p = p + (tf.math.square(g) - p) * (1.0 - tf.cast(beta_2, p.dtype))

      # Bias-corrected estimates
      m_hat = a / (1.0 - tf.math.pow(beta_1, casted_step))
      v_hat = p / (1.0 - tf.math.pow(beta_2, casted_step))

      # Compute ratio r = |m_hat| / max(|m_hat|, sqrt(v_hat)*epsilon)
      abs_m_hat = tf.abs(m_hat)
      max_abs_m_hat = tf.reduce_max(abs_m_hat)  # max(|m_hat|)
      r = abs_m_hat / (max_abs_m_hat * tf.cast(epsilon, abs_m_hat.dtype))

      # Compute lower and upper bounds for eta
      eta_l = tf.cast(alpha_star, r.dtype)
      eta_u = r + eta_l

      # Clip normalized_lr element-wise between eta_l and eta_u
      eta_candidate = tf.cast(normalized_lr, v_hat.dtype) / tf.sqrt(v_hat)
      eta = tf.clip_by_value(eta_candidate, eta_l, eta_u)

      # Print the values of eta_l, eta_u, and eta at runtime
      #tf.print("eta:", eta, summarize=-1)

      # Update weights
      w = w - m_hat * eta
      return w, a, p

    updated_weights, updated_accumulator, updated_preconditioner = (
        nest_utils.map_at_leaves(
            _adadb_update,
            weights,
            accumulator,
            preconditioner,
            gradients,
            num_outputs=3,
        )
    )

    updated_state = collections.OrderedDict([
        (optimizer.LEARNING_RATE_KEY, lr),
        (_BETA_1_KEY, beta_1),
        (_BETA_2_KEY, beta_2),
        (_EPSILON_KEY, epsilon),
        (_ALPHA_STAR_KEY, alpha_star),
        (_STEP_KEY, step),
        (_ACCUMULATOR_KEY, updated_accumulator),
        (_PRECONDITIONER_KEY, updated_preconditioner),
        (_CLIPNORM_KEY, clipnorm),
    ])
    return updated_state, updated_weights

  def get_hparams(self, state: State) -> Hparams:
    return collections.OrderedDict([(k, state[k]) for k in _HPARAMS_KEYS])

  def set_hparams(self, state: State, hparams: Hparams) -> State:
    return structure.update_struct(state, **hparams)


def build_custom_adadb(
  learning_rate: float,
  beta_1 = 0.9,
  beta_2 = 0.999,
  epsilon = 1e-7,
  alpha_star = 0.1,
  clipnorm = None, 
) -> optimizer.Optimizer:
  """Returns a `tff.learning.optimizers.Optimizer` for AdaDB.

    The AdaDB optimizer is an adaptive gradient method with data-dependent bounds.
    It builds upon Adam's bias-corrected moment estimates but introduces an upper 
    and lower bound on the element-wise learning rates based on current gradient 
    statistics. Specifically, it computes a ratio `r` influenced by the magnitude 
    of the bias-corrected first moments and a stability parameter `epsilon` (γ_t). 
    This ratio is then used to determine an upper and lower bound on the effective 
    per-parameter learning rate.
  
    In this scheme, the update is controlled by:
  
    Bias-corrected first and second moments:
    ```
    m_hat = acc / (1 - beta_1^t) v_hat = s / (1 - beta_2^t)
    Compute ratio r:
    
    r = |m_hat| / (max(|m_hat|) * epsilon)
    Determine bounds:
    
    eta_l = alpha_star eta_u = r + alpha_star
    Determine clipped learning rate:
    
    eta = clip(eta_l, (lr / sqrt(v_hat)), eta_u)
    Update rule:
    
    w = w - m_hat * eta
    ```
    
    Args:
      learning_rate: A positive `float` for the base learning rate.
      beta_1: A `float` in [0.0, 1.0], controlling the decay for the first moment.
      beta_2: A `float` in [0.0, 1.0], controlling the decay for the second moment.
      epsilon: A small non-negative `float` to prevent division by zero and 
        improve stability.
      alpha_star: A `float` for adjusting the lower bound of the clipped learning 
        rate range.
    
    Returns:
      A `tff.learning.optimizers.Optimizer` implementing the AdaDB update rule.
  """
  return _AdaDB(learning_rate, beta_1, beta_2, epsilon, alpha_star, clipnorm)
