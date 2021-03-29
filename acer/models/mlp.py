from typing import List, Callable, Tuple

import tensorflow as tf

from utils import normc_initializer


def build_mlp_network(layers_sizes: Tuple[int] = (256, 256), activation: str = 'tanh',
                      initializer: Callable = normc_initializer) \
        -> List[tf.keras.Model]:
    """Builds Multilayer Perceptron neural network

    Args:
        layers_sizes: sizes of hidden layers
        activation: activation function name
        initializer: callable to the weight initializer function

    Returns:
        created network
    """
    layers = [
        tf.keras.layers.Dense(
            layer_size,
            activation=activation,
            kernel_initializer=initializer()
        ) for layer_size in layers_sizes
    ]

    return layers
