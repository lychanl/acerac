"""
BaseActor-Critic with Experience Replay algorithm.
Implements the algorithm from:

(1)
Wawrzyński P, Tanwani AK. Autonomous reinforcement learning with experience replay.
Neural Networks : the Official Journal of the International Neural Network Society.
2013 May;41:156-167. DOI: 10.1016/j.neunet.2012.11.007.

(2)
Wawrzyński, Paweł. "Real-time reinforcement learning by sequential actor–critics
and experience replay." Neural Networks 22.10 (2009): 1484-1497.
"""
from typing import Optional, List, Union, Dict, Tuple
import gym
import tensorflow as tf
import numpy as np

from algos.base import BaseACERAgent, BaseActor, CategoricalActor, GaussianActor, Critic


class ACER(BaseACERAgent):
    def __init__(self, observations_space: gym.Space, actions_space: gym.Space, actor_layers: Optional[Tuple[int]],
                 critic_layers: Optional[Tuple[int]], lam: float = 0.1, b: float = 3, *args, **kwargs):
        """BaseActor-Critic with Experience Replay

        TODO: finish docstrings
        """

        super().__init__(observations_space, actions_space, actor_layers, critic_layers, *args, **kwargs)
        self._lam = lam
        self._b = b

    def _init_actor(self) -> BaseActor:
        if self._is_discrete:
            return CategoricalActor(
                self._observations_space, self._actions_space, self._actor_layers,
                self._actor_beta_penalty, self._tf_time_step
            )
        else:
            return GaussianActor(
                self._observations_space, self._actions_space, self._actor_layers,
                self._actor_beta_penalty, self._actions_bound, self._std, self._tf_time_step
            )

    def _init_critic(self) -> Critic:
        return Critic(self._observations_space, self._critic_layers, self._tf_time_step)

    def learn(self):
        """
        Performs experience replay learning. Experience trajectory is sampled from every replay buffer once, thus
        single backwards pass batch consists of 'num_parallel_envs' trajectories.

        Every call executes N of backwards passes, where: N = min(c0 * time_step / num_parallel_envs, c).
        That means at the beginning experience replay intensity increases linearly with number of samples
        collected till c value is reached.
        """
        if self._time_step > self._learning_starts:
            experience_replay_iterations = min([round(self._c0 * self._time_step), self._c])
            
            for batch in self._data_loader.take(experience_replay_iterations):
                self._learn_from_experience_batch(*batch)

    @tf.function(experimental_relax_shapes=True)
    def _learn_from_experience_batch(self, obs, obs_next, actions, old_policies,
                                     rewards, first_obs, first_actions, dones, lengths):
        """Backward pass with single batch of experience.

        Every experience replay requires sequence of experiences with random length, thus we have to use
        ragged tensors here.

        See Equation (8) and Equation (9) in the paper (1).
        """

        obs = self._process_observations(obs)
        obs_next = self._process_observations(obs_next)
        rewards = self._process_rewards(rewards)

        batches_indices = tf.RaggedTensor.from_row_lengths(values=tf.range(tf.reduce_sum(lengths)), row_lengths=lengths)
        values = tf.squeeze(self._critic.value(obs))
        values_next = tf.squeeze(self._critic.value(obs_next)) * (1.0 - tf.cast(dones, tf.dtypes.float32))
        policies, log_policies = tf.split(self._actor.prob(obs, actions), 2, axis=0)
        policies, log_policies = tf.squeeze(policies), tf.squeeze(log_policies)
        indices = tf.expand_dims(batches_indices, axis=2)

        # flat tensor
        policies_ratio = tf.math.divide(policies, old_policies)
        # ragged tensor divided into batches
        policies_ratio_batches = tf.squeeze(tf.gather(policies_ratio, indices), axis=2)

        # cumprod and cumsum do not work on ragged tensors, we transform them into tensors
        # padded with 0 and then apply boolean mask to retrieve original ragged tensor
        batch_mask = tf.sequence_mask(policies_ratio_batches.row_lengths())
        policies_ratio_product = tf.math.cumprod(policies_ratio_batches.to_tensor(), axis=1)

        truncated_densities = tf.ragged.boolean_mask(
            tf.minimum(policies_ratio_product, self._b),
            batch_mask
        )
        gamma_coeffs_batches = tf.ones_like(truncated_densities).to_tensor() * self._gamma
        gamma_coeffs = tf.ragged.boolean_mask(
            tf.math.cumprod(gamma_coeffs_batches, axis=1, exclusive=True),
            batch_mask
        ).flat_values

        # flat tensors
        d_coeffs = gamma_coeffs * (rewards + self._gamma * values_next - values) * truncated_densities.flat_values
        # ragged
        d_coeffs_batches = tf.gather_nd(d_coeffs, tf.expand_dims(indices, axis=2))
        # final summation over original batches
        d = tf.stop_gradient(tf.reduce_sum(d_coeffs_batches, axis=1))

        self._backward_pass(first_obs, first_actions, d)

        _, new_log_policies = tf.split(self._actor.prob(obs, actions), 2, axis=0)
        new_log_policies = tf.squeeze(new_log_policies)
        approx_kl = tf.reduce_mean(policies - new_log_policies)
        with tf.name_scope('actor'):
            tf.summary.scalar('sample_approx_kl_divergence', approx_kl, self._tf_time_step)

    def _backward_pass(self, observations: tf.Tensor, actions: tf.Tensor, d: tf.Tensor):
        """Performs backward pass in BaseActor's and Critic's networks

        Args:
            observations: batch [batch_size, observations_dim] of observations vectors
            actions: batch [batch_size, actions_dim] of actions vectors
            d: batch [batch_size, observations_dim] of gradient update coefficient
                (summation terms in the Equations (8) and (9) from the paper (1))
        """
        with tf.GradientTape() as tape:
            loss = self._actor.loss(observations, actions, d)
        grads = tape.gradient(loss, self._actor.trainable_variables)
        if self._gradient_norm is not None:
            grads = self._clip_gradient(grads, self._actor_gradient_norm_median, 'actor')
        gradients = zip(grads, self._actor.trainable_variables)

        self._actor_optimizer.apply_gradients(gradients)

        with tf.GradientTape() as tape:
            loss = self._critic.loss(observations, d)
        grads = tape.gradient(loss, self._critic.trainable_variables)
        if self._gradient_norm is not None:
            grads = self._clip_gradient(grads, self._critic_gradient_norm_median, 'critic')
        gradients = zip(grads, self._critic.trainable_variables)

        self._critic_optimizer.apply_gradients(gradients)

    def _fetch_offline_batch(self) -> List[Dict[str, Union[np.array, list]]]:
        trajectory_lens = [np.random.geometric(1 - self._lam) + 1 for _ in range(self._num_parallel_envs)]
        batch = []
        [batch.extend(self._memory.get(trajectory_lens)) for _ in range(self._batches_per_env)]
        return batch
