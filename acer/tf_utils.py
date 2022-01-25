from typing import Union, Tuple

import numpy as np
import tensorflow as tf


def kronecker_prod(x: Union[tf.Tensor, np.array], y: Union[tf.Tensor, np.array]) -> tf.Tensor:
    """Computes Kronecker product between x and y tensors

    Args:
        x: first tensor
        y: second tensor

    Returns:
        x and y product
    """
    operator_1 = tf.linalg.LinearOperatorFullMatrix(x)
    operator_2 = tf.linalg.LinearOperatorFullMatrix(y)
    prod = tf.linalg.LinearOperatorKronecker([operator_1, operator_2]).to_dense()
    return prod


class RunningMeanVarianceTf:
    def __init__(self, epsilon: float = 1e-4, shape: Tuple = ()):
        """TensorFlow version of RunningMeanVariance

        Args:
            epsilon: small value for numerical stability
            shape: shape of the normalized vector
        """
        self.mean = tf.Variable(initial_value=tf.zeros(shape=shape, dtype=tf.float32), trainable=False)
        self.var = tf.Variable(initial_value=tf.ones(shape=shape, dtype=tf.float32), trainable=False)
        self.count = tf.Variable(initial_value=epsilon, trainable=False, dtype=tf.float32)

    @tf.function(experimental_relax_shapes=True)
    def update(self, x: tf.Tensor):
        """Updates statistics with given batch [batch_size, vector_size] of samples

        Args:
            x: batch of samples
        """
        batch_mean = tf.reduce_mean(x, axis=0)
        batch_var = tf.math.reduce_variance(x, axis=0)
        batch_count = x.shape[0]

        if tf.math.less(self.count, 1):
            self._assign_new_values(batch_count, batch_mean, batch_var)
        else:
            new_count = self.count + batch_count
            delta = batch_mean - self.mean
            new_mean = self.mean + delta * batch_count / new_count

            m_a = self.var * (self.count - 1.0)
            m_b = batch_var * (batch_count - 1.0)
            m_2 = m_a + m_b + tf.square(delta) * self.count * batch_count / new_count
            new_var = m_2 / (new_count - 1.0)
            self._assign_new_values(new_count, new_mean, new_var)

    def _assign_new_values(self, count: tf.Tensor, mean: tf.Tensor, var: tf.Tensor):
        self.count.assign(count)
        self.mean.assign(mean)
        self.var.assign(var)

    def save(self, path: str):
        """Saves the state on disk"""
        pass

    def load(self, path: str):
        """Load the state from disk"""
        pass
