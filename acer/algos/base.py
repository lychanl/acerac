from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple, Union, List, Optional, Dict

import gym
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import tf_utils
import utils
from models.mlp import build_mlp_network
from replay_buffer import MultiReplayBuffer, BufferFieldSpec, ReplayBuffer

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


class BaseActor(ABC, tf.keras.Model):

    def __init__(self, observations_space: gym.Space, actions_space: gym.Space, layers: Optional[Tuple[int]],
                 beta_penalty: float, tf_time_step: tf.Variable, *args, **kwargs):
        """Base abstract Actor class

        Args:
            observations_dim: dimension of observations space
            layers: list of hidden layers sizes, eg: for neural network with two layers with 10 and 20 hidden units
                pass: [10, 20]
            beta_penalty: penalty coefficient. In discrete case, BaseActor is penalized for too executing too
                confident actions (no exploration), in the continuous case it is penalized for making actions
                that are out of the allowed bounds
            tf_time_step: time step as TensorFlow variable, required for TensorBoard summaries
        """
        super().__init__(*args, **kwargs)

        self._hidden_layers = []

        if type(actions_space) == gym.spaces.discrete.Discrete:
            actions_dim = actions_space.n
        else:
            actions_dim = actions_space.shape[0]
        
        self._hidden_layers.extend(build_mlp_network(layers_sizes=layers))

        self._hidden_layers.append(tf.keras.layers.Dense(actions_dim, kernel_initializer=utils.normc_initializer()))

        self.actions_dim = actions_dim
        self.beta_penalty = beta_penalty
        self._tf_time_step = tf_time_step

    def _forward(self, observations: np.array) -> tf.Tensor:
        x = self._hidden_layers[0](observations)
        for layer in self._hidden_layers[1:]:
            x = layer(x)
        return x

    @property
    @abstractmethod
    def action_dtype(self):
        """Returns data type of the BaseActor's actions (TensorFlow)"""

    @property
    @abstractmethod
    def action_dtype_np(self):
        """Returns data type of the BaseActor's actions (Numpy)"""

    @abstractmethod
    def prob(self, observations: np.array, actions: np.array) -> tf.Tensor:
        """Computes probabilities (or probability densities in continuous case) and logarithms of it

        Args:
            observations: batch [batch_size, observations_dim] of observations vectors
            actions: batch [batch_size, actions_dim] of actions vectors

        Returns:
             Tensor [batch_size, actions_dim, 2] with computed probabilities (densities) and logarithms
        """

    @abstractmethod
    def act(self, observations: np.array, **kwargs) -> Tuple[tf.Tensor, tf.Tensor]:
        """Samples actions and computes their probabilities (or probability densities in continuous case)

        Args:
            observations: batch [batch_size, observations_dim] of observations vectors

        Returns:
            tuple with two Tensors:
                * actions [batch_size, actions_dim]
                * probabilities/densities [batch_size, 1]
        """

    @abstractmethod
    def act_deterministic(self, observations: np.array, **kwargs) -> tf.Tensor:
        """Samples actions without exploration noise.

        Args:
            observations: batch [batch_size, observations_dim] of observations vectors

        Returns:
            Tensor of actions [batch_size, actions_dim]
        """


class BaseCritic(ABC, tf.keras.Model):
    def __init__(self, observations_space: gym.Space, layers: Optional[Tuple[int]],
                 tf_time_step: tf.Variable, use_additional_input: bool = False, *args, **kwargs):
        """Value function approximation as MLP network neural network.

        Args:
            observations_dim: dimension of observations space
            layers: list of hidden layers sizes, eg: for neural network with two layers with 10 and 20 hidden units
                pass: [10, 20]
            tf_time_step: time step as TensorFlow variable, required for TensorBoard summaries
            additional_input_shape: shape of additional input variables
        """
        super().__init__(*args, **kwargs)
        self._hidden_layers = []

        self._hidden_layers.extend(build_mlp_network(layers_sizes=layers))

        self._v = tf.keras.layers.Dense(1, kernel_initializer=utils.normc_initializer())
        self._tf_time_step = tf_time_step
        self._use_additional_input = use_additional_input

    def call(self, inputs, training=None, mask=None, additional_input=None):
        return self.value(inputs, additional_input=additional_input)

    def value(self, observations: tf.Tensor, additional_input: tf.Tensor=None, **kwargs) -> tf.Tensor:
        """Calculates value function given observations batch

        Args:
            observations: batch [batch_size, observations_dim] of observations vectors


        Returns:
            Tensor [batch_size, 1] with value function estimations

        """
        x = tf.concat([observations, additional_input], axis=-1) if self._use_additional_input else observations

        for layer in self._hidden_layers:
            x = layer(x)

        value = self._v(x)

        return value


class Critic(BaseCritic):

    def __init__(self, observations_space: gym.Space, layers: Optional[Tuple[int]], tf_time_step: tf.Variable,
                 use_additional_input: bool = False, *args, **kwargs):
        """Basic Critic that outputs single value"""
        super().__init__(observations_space, layers, tf_time_step, *args, use_additional_input=use_additional_input, **kwargs)

    def loss(self, observations: np.array, d: np.array, additional_input=None) -> tf.Tensor:
        """Computes Critic's loss.

        Args:
            observations: batch [batch_size, observations_dim] of observations vectors
            d: batch [batch_size, 1] of gradient update coefficient (summation term in the Equation (9)) from
                the paper (1))
        """

        value = self.value(observations, additional_input=additional_input)
        loss = tf.reduce_mean(-tf.math.multiply(value, d))

        with tf.name_scope('critic'):
            tf.summary.scalar('batch_value_mean', tf.reduce_mean(value), step=self._tf_time_step)
            tf.summary.scalar('batch_loss', loss, step=self._tf_time_step)
        return loss


class CategoricalActor(BaseActor):

    def __init__(self, observations_space: gym.Space, actions_space: gym.Space, layers: Optional[Tuple[int]],
                 *args, **kwargs):
        """BaseActor for discrete actions spaces. Uses Categorical Distribution"""
        super().__init__(observations_space, actions_space, layers, *args, **kwargs)

    @property
    def action_dtype(self):
        return tf.dtypes.int32

    @property
    def action_dtype_np(self):
        return np.int32

    def loss(self, observations: tf.Tensor, actions: tf.Tensor, d: tf.Tensor) -> tf.Tensor:
        logits = self._forward(observations)

        # TODO: remove hardcoded '10' and '20'
        logits_div = tf.divide(logits, 10)
        log_probs = tf.nn.log_softmax(logits_div)
        action_log_probs = tf.expand_dims(
            tf.gather_nd(log_probs, actions, batch_dims=1),
            axis=1
        )
        dist = tfp.distributions.Categorical(logits_div)

        penalty = tf.reduce_sum(
            tf.scalar_mul(
                self.beta_penalty,
                tf.square(tf.maximum(0.0, tf.abs(logits) - 20))
            ),
            axis=1,
            keepdims=True
        )
        total_loss = tf.reduce_mean(-tf.math.multiply(action_log_probs, d) + penalty)

        # entropy maximization penalty
        # entropy = -tf.reduce_sum(tf.math.multiply(probs, log_probs), axis=1)
        # penalty = self.beta_penalty * (-tf.reduce_sum(tf.math.multiply(probs, log_probs), axis=1))

        with tf.name_scope('actor'):
            tf.summary.scalar('batch_entropy_mean', tf.reduce_mean(dist.entropy()), step=self._tf_time_step)
            tf.summary.scalar('batch_loss', total_loss, step=self._tf_time_step)
            tf.summary.scalar('batch_penalty_mean', tf.reduce_mean(penalty), step=self._tf_time_step)

        return total_loss

    def prob(self, observations: tf.Tensor, actions: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        # TODO: remove hardcoded '10' and '20'
        logits = tf.divide(self._forward(observations), 10)
        probs = tf.nn.softmax(logits)
        log_probs = tf.nn.log_softmax(logits)
        action_probs = tf.gather_nd(probs, actions, batch_dims=1)
        action_log_probs = tf.gather_nd(log_probs, actions, batch_dims=1)
        return action_probs, action_log_probs

    @tf.function
    def act(self, observations: tf.Tensor, **kwargs) -> Tuple[tf.Tensor, tf.Tensor]:

        # TODO: remove hardcoded '10' and '20'
        logits = tf.divide(self._forward(observations), 10)
        probs = tf.nn.softmax(logits)
        log_probs = tf.nn.log_softmax(logits)

        actions = tf.random.categorical(log_probs, num_samples=1, dtype=tf.dtypes.int32)
        actions_probs = tf.gather_nd(probs, actions, batch_dims=1)

        with tf.name_scope('actor'):
            # TODO: refactor
            tf.summary.histogram('action', actions, step=self._tf_time_step)
        return tf.squeeze(actions, axis=[1]), actions_probs

    @tf.function
    def act_deterministic(self, observations: tf.Tensor, **kwargs) -> tf.Tensor:
        """Performs most probable action"""
        logits = tf.divide(self._forward(observations), 10)
        probs = tf.nn.softmax(logits)

        actions = tf.argmax(probs, axis=1)
        return actions


class GaussianActor(BaseActor):

    def __init__(self, observations_space: gym.Space, actions_space: gym.Space, layers: Optional[Tuple[int]],
                 beta_penalty: float, actions_bound: float, std: float = None, *args, **kwargs):
        """BaseActor for continuous actions space. Uses MultiVariate Gaussian Distribution as policy distribution.

        TODO: introduce [a, b] intervals as allowed actions bounds

        Args:
            observations_dim: dimension of observations space
            layers: list of hidden layer sizes
            beta_penalty: penalty for too confident actions coefficient
            actions_bound: upper (lower == '-actions_bound') bound for allowed actions,
             required in case of continuous actions
        """
        super().__init__(observations_space, actions_space, layers, beta_penalty, *args, **kwargs)

        self._actions_bound = actions_bound

        if std:
            # change constant to Variable to make std a learned parameter
            self.log_std = tf.constant(
                tf.math.log([std] * actions_space.shape[0]),
                name="actor_std",
            )
        else:
            self.log_std = tf.constant(
                tf.math.log(0.4 * actions_bound),
                name="actor_std",
            )

    @property
    def action_dtype(self):
        return tf.dtypes.float32

    @property
    def action_dtype_np(self):
        return np.float32

    def loss(self, observations: np.array, actions: np.array, d: np.array) -> tf.Tensor:
        mean = self._forward(observations)
        dist = tfp.distributions.MultivariateNormalDiag(
            loc=mean,
            scale_diag=tf.exp(self.log_std)
        )

        action_log_probs = tf.expand_dims(dist.log_prob(actions), axis=1)

        bounds_penalty = tf.reduce_sum(
            tf.scalar_mul(
                self.beta_penalty,
                tf.square(tf.maximum(0.0, tf.abs(mean) - self._actions_bound))
            ),
            axis=1,
            keepdims=True
        )
        entropy = dist.entropy()
        # entropy_penalty = 0.01 * entropy

        total_loss = tf.reduce_mean(-tf.math.multiply(action_log_probs, d) + bounds_penalty)

        with tf.name_scope('actor'):
            for i in range(self.actions_dim):
                tf.summary.scalar(f'std_{i}', tf.exp(self.log_std[i]), step=self._tf_time_step)
            tf.summary.scalar('batch_loss', total_loss, step=self._tf_time_step)
            tf.summary.scalar('batch_bounds_penalty_mean', tf.reduce_mean(bounds_penalty), step=self._tf_time_step)
            tf.summary.scalar('batch_entropy_mean', tf.reduce_mean(entropy), step=self._tf_time_step)

        return total_loss

    def prob(self, observations: tf.Tensor, actions: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        mean = self._forward(observations)
        dist = tfp.distributions.MultivariateNormalDiag(
            loc=mean,
            scale_diag=tf.exp(self.log_std)
        )

        return dist.prob(actions), dist.log_prob(actions)

    @tf.function
    def act(self, observations: tf.Tensor, **kwargs) -> Tuple[tf.Tensor, tf.Tensor]:
        mean = self._forward(observations)

        dist = tfp.distributions.MultivariateNormalDiag(
            loc=mean,
            scale_diag=tf.exp(self.log_std)
        )

        actions = dist.sample(dtype=self.dtype)
        actions_probs = dist.prob(actions)

        with tf.name_scope('actor'):
            tf.summary.scalar(f'batch_action_mean', tf.reduce_mean(actions), step=self._tf_time_step)

        return actions, actions_probs

    @tf.function
    def act_deterministic(self, observations: tf.Tensor, **kwargs) -> tf.Tensor:
        """Returns mean of the Gaussian"""
        mean = self._forward(observations)
        return mean


class BaseACERAgent(ABC):
    """Base ACER abstract class"""
    def __init__(self, observations_space: gym.Space, actions_space: gym.Space, actor_layers: Optional[Tuple[int]],
                 critic_layers: Tuple[int], gamma: int = 0.99, actor_beta_penalty: float = 0.001,
                 std: Optional[float] = None, memory_size: int = 1e6, num_parallel_envs: int = 10,
                 batches_per_env: int = 5, c: int = 10, c0: float = 0.3, actor_lr: float = 0.001,
                 actor_adam_beta1: float = 0.9, actor_adam_beta2: float = 0.999, actor_adam_epsilon: float = 1e-7,
                 critic_lr: float = 0.001, critic_adam_beta1: float = 0.9, critic_adam_beta2: float = 0.999,
                 critic_adam_epsilon: float = 1e-7, standardize_obs: bool = False, rescale_rewards: int = -1,
                 limit_reward_tanh: float = None, time_step: int = 1, gradient_norm: float = None,
                 gradient_norm_median_threshold: float = 4, learning_starts: int = 1000, **kwargs):

        self._tf_time_step = tf.Variable(
            initial_value=time_step, name='tf_time_step', dtype=tf.dtypes.int64, trainable=False
        )
        self._observations_space = observations_space
        self._actions_space = actions_space
        self._std = std
        self._actor_beta_penalty = actor_beta_penalty
        self._c = c
        self._c0 = c0
        self._learning_starts = learning_starts
        self._actor_layers = tuple(actor_layers)
        self._critic_layers = tuple(critic_layers)
        self._gamma = gamma
        self._batches_per_env = batches_per_env
        self._time_step = 0
        self._num_parallel_envs = num_parallel_envs
        self._limit_reward_tanh = limit_reward_tanh
        self._gradient_norm = gradient_norm
        self._gradient_norm_median_threshold = gradient_norm_median_threshold
        self._batch_size = self._num_parallel_envs * self._batches_per_env

        self._actor_gradient_norm_median = tf.Variable(initial_value=1.0, trainable=False)
        self._critic_gradient_norm_median = tf.Variable(initial_value=1.0, trainable=False)

        if type(actions_space) == gym.spaces.Discrete:
            self._is_discrete = True
            self._actions_bound = 0
        else:
            self._is_discrete = False
            self._actions_bound = actions_space.high

        self._actor = self._init_actor()
        self._critic = self._init_critic()

        self._init_replay_buffer(memory_size)
        self._data_loader = tf.data.Dataset.from_generator(
            self._experience_replay_generator,
            (tf.dtypes.float32, tf.dtypes.float32, self._actor.action_dtype, tf.dtypes.float32, tf.dtypes.float32,
             tf.dtypes.float32, self._actor.action_dtype, tf.dtypes.bool, tf.dtypes.int32)
        ).prefetch(1)

        self._actor_optimizer = tf.keras.optimizers.Adam(
            lr=actor_lr,
            beta_1=actor_adam_beta1,
            beta_2=actor_adam_beta2,
            epsilon=actor_adam_epsilon
        )

        self._critic_optimizer = tf.keras.optimizers.Adam(
            lr=critic_lr,
            beta_1=critic_adam_beta1,
            beta_2=critic_adam_beta2,
            epsilon=critic_adam_epsilon
        )

        if standardize_obs:
            self._running_mean_obs = tf_utils.RunningMeanVarianceTf(shape=observations_space.shape)
        else:
            self._running_mean_obs = None

        self._rescale_rewards = rescale_rewards
        if rescale_rewards == 0:
            self._running_mean_rewards = tf_utils.RunningMeanVarianceTf(shape=(1, ))
        else:
            self._running_mean_rewards = None

    def _init_replay_buffer(self, memory_size: int):
        if type(self._actions_space) == gym.spaces.Discrete:
            actions_shape = (1, )
        else:
            actions_shape = self._actions_space.shape

        self._memory = MultiReplayBuffer(
            buffer_class=ReplayBuffer,
            action_spec=BufferFieldSpec(shape=actions_shape, dtype=self._actor.action_dtype_np),
            obs_spec=BufferFieldSpec(shape=self._observations_space.shape, dtype=self._observations_space.dtype),
            max_size=memory_size,
            num_buffers=self._num_parallel_envs
        )

    def _clip_gradient(
            self, grads: List[tf.Tensor], norm_variable: tf.Variable, scope: str
    ) -> List[tf.Tensor]:
        if self._gradient_norm == 0:
            grads_clipped, grads_norm = tf.clip_by_global_norm(
                grads,
                norm_variable * self._gradient_norm_median_threshold
            )
            update_sign = tf.pow(
                -1.0, tf.cast(tf.less(grads_norm, norm_variable), dtype=tf.float32)
            )
            norm_variable.assign_add(
                update_sign * norm_variable * 0.01
            )
            with tf.name_scope(scope):
                tf.summary.scalar("gradient_norm_median", norm_variable, self._tf_time_step)
        else:
            grads_clipped, grads_norm = tf.clip_by_global_norm(
                grads,
                self._gradient_norm
            )
        with tf.name_scope(scope):
            tf.summary.scalar("gradient_norm", grads_norm, self._tf_time_step)
        return grads_clipped

    def save_experience(self, steps: List[
        Tuple[Union[int, float, list], np.array, float, np.array, bool, bool]
    ]):
        """Stores gathered experiences in a replay buffer. Accepts list of steps.

        Args:
            steps: List of steps, see ReplayBuffer.put() for a detailed format description
        """
        self._time_step += len(steps)
        self._tf_time_step.assign_add(len(steps))
        self._memory.put(steps)

        if self._running_mean_obs:
            obs = np.array([step[1] for step in steps])
            self._update_obs_rms(obs)
        if self._running_mean_rewards:
            rewards = np.array([step[2] for step in steps])
            self._update_rewards_rms(rewards)

    @tf.function(experimental_relax_shapes=True)
    def _update_obs_rms(self, obs):
        if self._running_mean_obs:
            self._running_mean_obs.update(obs)
            with tf.name_scope('observations'):
                tf.summary.scalar(
                    f'obs_running_mean_norm',
                    tf.linalg.norm(self._running_mean_obs.mean),
                    step=self._tf_time_step
                )
                tf.summary.scalar(
                    f'obs_running_std_norm',
                    tf.linalg.norm(tf.sqrt(self._running_mean_obs.var)),
                    step=self._tf_time_step
                )

    @tf.function(experimental_relax_shapes=True)
    def _update_rewards_rms(self, rewards):
        if self._running_mean_rewards:
            self._running_mean_rewards.update(tf.expand_dims(tf.cast(rewards, dtype=tf.float32), axis=1))
            with tf.name_scope('rewards'):
                tf.summary.scalar(
                    f'rewards_running_std', tf.sqrt(self._running_mean_rewards.var[0]), step=self._tf_time_step
                )

    def predict_action(self, observations: np.array, is_deterministic: bool = False) \
            -> Tuple[np.array, Optional[np.array]]:
        """Predicts actions for given observations. Performs forward pass with BaseActor network.

        Args:
            observations: batch [batch_size, observations_dim] of observations vectors
            is_deterministic: True if mean actions (without exploration noise) should be returned

        Returns:
            Tuple of sampled actions and corresponding probabilities (probability densities) if action was sampled
                from the distribution, None otherwise
        """
        processed_obs = tf.convert_to_tensor(self._process_observations(observations))
        if is_deterministic:
            return self._actor.act_deterministic(processed_obs).numpy(), None
        else:
            actions, policies = self._actor.act(processed_obs)
            return actions.numpy(), policies.numpy()

    def _experience_replay_generator(self):
        while True:
            offline_batch = self._fetch_offline_batch()

            obs_flatten, obs_next_flatten, actions_flatten, policies_flatten, rewards_flatten, dones_flatten \
                = utils.flatten_experience(offline_batch)

            lengths = [len(batch[0]['observations']) for batch in offline_batch]

            first_obs = [batch[0]['observations'][0] for batch in offline_batch]
            first_actions = [batch[0]['actions'][0] for batch in offline_batch]

            yield (
                obs_flatten,
                obs_next_flatten,
                actions_flatten,
                policies_flatten,
                rewards_flatten,
                first_obs,
                first_actions,
                dones_flatten,
                lengths
            )

    def _process_observations(self, observations: tf.Tensor) -> tf.Tensor:
        """If standardization is turned on, observations are being standardized with running mean and variance.
        Additional clipping is used to prevent performance spikes."""
        if self._running_mean_obs:
            centered = observations - self._running_mean_obs.mean
            standardized = centered / tf.sqrt(self._running_mean_obs.var + 1e-8)
            return tf.clip_by_value(
                standardized,
                -10.0,
                10.0
            )
        else:
            return observations

    def _process_rewards(self, rewards: tf.Tensor) -> tf.Tensor:
        """Rescales returns with standard deviation. Additional clipping is used to prevent performance spikes."""
        if self._rescale_rewards == 0:
            rewards = rewards / tf.sqrt(self._running_mean_rewards.var + 1e-8)
            if not self._limit_reward_tanh:
                rewards = tf.clip_by_value(
                    rewards / tf.sqrt(self._running_mean_rewards.var + 1e-8),
                    -5.0,
                    5.0
                )
        elif self._rescale_rewards >= 0:
            rewards = rewards / self._rescale_rewards

        if self._limit_reward_tanh:
            rewards = tf.tanh(rewards / self._limit_reward_tanh) * self._limit_reward_tanh

        return rewards

    @abstractmethod
    def _fetch_offline_batch(self) -> List[Tuple[Dict[str, Union[np.array, list]], int]]:
        ...

    @abstractmethod
    def learn(self):
        ...

    @abstractmethod
    def _init_actor(self) -> BaseActor:
        ...

    @abstractmethod
    def _init_critic(self) -> BaseCritic:
        ...

    def save(self, path: Path, **kwargs):
        actor_path = str(path / 'actor.tf')
        critic_path = str(path / 'critic.tf')
        buffer_path = str(path / 'buffer.pkl')

        self._actor.save_weights(actor_path, overwrite=True)
        self._critic.save_weights(critic_path, overwrite=True)

        if self._running_mean_obs:
            rms_obs_path = str(path / 'rms_obs.pkl')
            self._running_mean_obs.save(rms_obs_path)

        if self._running_mean_rewards:
            rms_rewards_path = str(path / 'rms_rewards.pkl')
            self._running_mean_rewards.save(rms_rewards_path)

        self._memory.save(buffer_path)

    def load(self, path: Path, **kwargs):
        actor_path = str(path / 'actor.tf')
        critic_path = str(path / 'critic.tf')
        buffer_path = str(path / 'buffer.pkl')

        self._actor.load_weights(actor_path, overwrite=True)
        self._critic.load_weights(critic_path, overwrite=True)

        if self._running_mean_obs:
            rms_obs_path = str(path / 'rms_obs.pkl')
            self._running_mean_obs.load(rms_obs_path)

        if self._running_mean_rewards:
            rms_rewards_path = str(path / 'rms_rewards.pkl')
            self._running_mean_rewards.load(rms_rewards_path)

        self._memory.load(buffer_path)
