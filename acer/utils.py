import pickle
from functools import partial, wraps
from time import time
from typing import Tuple, List, Union, Dict
# import importlib
import re

import gym
import numpy as np
from numpy.typing import NDArray
import tensorflow as tf


def normc_initializer():
    """Normalized column initializer from the OpenAI baselines"""

    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape)
        out *= 1 / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out, dtype=dtype)

    return _initializer


def flatten_experience(buffers_batches: List[Tuple[Dict[str, Union[np.array, list]], int]])\
        -> Tuple[np.array, np.array, np.array, np.array, np.array, np.array]:
    """Parses experience from the buffers (from dictionaries) into matrices that can be feed into
    neural network in a single pass.

    Args:
        buffers_batches: trajectories fetched from replay buffers

    Returns:
        Tuple with matrices:
            * batch [batch_size, observations_dim] of observations
            * batch [batch_size, observations_dim] of 'next' observations
            * batch [batch_size, actions_dim] of actions
    """
    observations = np.concatenate([batch[0]['observations'] for batch in buffers_batches], axis=0)
    next_observations = np.concatenate([batch[0]['next_observations'] for batch in buffers_batches], axis=0)
    actions = np.concatenate([batch[0]['actions'] for batch in buffers_batches], axis=0)
    policies = np.concatenate([batch[0]['policies'] for batch in buffers_batches], axis=0)
    rewards = np.concatenate([batch[0]['rewards'] for batch in buffers_batches], axis=0)
    dones = np.concatenate([batch[0]['dones'] for batch in buffers_batches], axis=0)

    return observations, next_observations, actions, policies, rewards, dones


def is_atari(env_id: str) -> bool:
    """Checks if environments if of Atari type
    Args:
        env_id: name of the environment
    Returns:
        True if its is Atari env
    """
    env_specs = [env for env in gym.envs.registry.all() if env.id == env_id]
    if not env_specs:
        return False
    env_spec = env_specs[0]

    if not isinstance(env_spec.entry_point, str):
        return False
    env_type = env_spec.entry_point.split(':')[0].split('.')[-1]
    return env_type == 'atari'


def get_env_variables(env):
    """Returns OpenAI Gym environment specific variables like action space dimension"""
    if type(env.observation_space) == gym.spaces.discrete.Discrete:
        observations_dim = env.observation_space.n
    else:
        observations_dim = env.observation_space.shape[0]
    if type(env.action_space) == gym.spaces.discrete.Discrete:
        continuous = False
        actions_dim = env.action_space.n
        action_scale = 1
    else:
        continuous = True
        actions_dim = env.action_space.shape[0]
        action_scale = np.maximum(env.action_space.high, np.abs(env.action_space.low))
    max_steps_in_episode = env.spec.max_episode_steps
    return action_scale, actions_dim, observations_dim, continuous, max_steps_in_episode


class RunningMeanVariance:

    def __init__(self, epsilon: float = 1e-4, shape: Tuple = ()):
        """Computes running mean and variance with Welford's online algorithm (Parallel algorithm)

        Reference:
            https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance

        Args:
            epsilon: small value for numerical stability
            shape: shape of the normalized vector
        """
        self.mean = np.zeros(shape=shape, dtype=np.float32)
        self.var = np.ones(shape=shape, dtype=np.float32)
        self.count = epsilon

    def update(self, x: NDArray):
        """Updates statistics with given batch [batch_size, vector_size] of samples

        Args:
            x: batch of samples
        """
        batch_mean = np.mean(x, axis=0, dtype=np.float32)
        batch_var = np.var(x, axis=0, dtype=np.float32)
        batch_count = x.shape[0]

        if self.count < 1:
            self.count, self.mean, self.var = batch_count, batch_mean, batch_var
        else:
            new_count = self.count + batch_count
            delta = batch_mean - self.mean
            new_mean = self.mean + delta * batch_count / new_count

            m_a = self.var * (self.count - 1)
            m_b = batch_var * (batch_count - 1)
            m_2 = m_a + m_b + np.square(delta) * self.count * batch_count / new_count
            new_var = m_2 / (new_count - 1)
            self.count, self.mean, self.var = new_count, new_mean, new_var

    def save(self, path: str):
        """Saves the state on disk"""
        with open(path, 'wb') as f:
            pickle.dump(self, f)


def timing(f):
    """Function decorator that measures time elapsed while executing a function."""

    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('func:%r took: %2.4f sec' % (f.__name__, te - ts))
        return result

    return wrap


def getDTChangedEnvName(base_env_name, timesteps_increase):
    return str.join('TS' + str(timesteps_increase) + '-', base_env_name.split('-'))


def getPossiblyDTChangedEnvBuilder(env_name):
    prog = re.compile(r'(\w+)TS(\d+)\-v(\d+)')
    match = prog.fullmatch(env_name)
    if not match:
        return partial(gym.make, env_name)

    name = match.group(1)
    timesteps_increase = int(match.group(2))
    version = match.group(3)

    base_name = f'{name}-v{version}'

    base_spec = gym.envs.registry.env_specs[base_name]

    # mod_name, _ = base_spec.entry_point.split(":")
    # importlib.import_module(mod_name)

    def builder():
        env = gym.make(base_name)

        if base_spec.max_episode_steps:
            env = gym.wrappers.TimeLimit(gym.wrappers.TransformReward(env, lambda r: r / 3),
                                         base_spec.max_episode_steps * timesteps_increase)

        return env

    return builder


def calculate_gamma(gamma0, timesteps_increase):
    return gamma0**(1 / timesteps_increase)
