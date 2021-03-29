from collections import deque
from typing import Optional, List, Union, Dict, Tuple
# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import gym
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

import tf_utils
from algos.base import BaseActor, Critic, BaseACERAgent, GaussianActor
from replay_buffer import BufferFieldSpec, PrevReplayBuffer, MultiReplayBuffer


def get_lambda_1(n: int, alpha: float) -> np.array:
    """Computes Lambda^n_1 matrix.

    Args:
        n: size of the matrix
        alpha: autocorrelation coefficient

    Returns:
        Lambda^n_1 matrix
    """
    lam = np.zeros(shape=(n + 1, n + 1), dtype=np.float32)
    for i in range(n + 1):
        for j in range(i, n + 1):
            lam[i][j] = lam[j][i] = alpha ** abs(i - j) - alpha ** (i + j + 2)
    return lam


def get_lambda_0(n: int, alpha: float) -> np.array:
    """Computes Lambda^n_0 matrix.

    Args:
        n: size of the matrix
        alpha: autocorrelation coefficient

    Returns:
        Lambda^n_1 matrix
    """
    lam = np.zeros(shape=(n + 1, n + 1), dtype=np.float32)
    for i in range(n + 1):
        for j in range(i, n + 1):
            lam[i][j] = lam[j][i] = alpha ** abs(i - j)
    return lam


class AutocorrelatedActor(GaussianActor):
    def __init__(self, *args, required_prev_samples, **kwargs):
        super().__init__(*args, **kwargs)
        self.required_prev_samples = required_prev_samples
        self._cov_matrix = tf.linalg.diag(tf.square(tf.exp(self.log_std)))
    
    def init_inverse_covariance_matrices(self, n: int, batch_size: int) -> Tuple[tf.Tensor, tf.Tensor]:
        raise NotImplementedError()
    
    def calculate_q_noise(self, actions: tf.Tensor, actor_outs: tf.Tensor, lengths: tf.Tensor,
            prev_actions: tf.Tensor, prev_actor_outs: tf.Tensor, prev_samples_len: tf.Tensor, actor_outs_next: tf.Tensor):
        raise NotImplementedError()
    
    def estimate_noise_from_prev(self, actions: tf.Tensor, prev_samples_len: tf.Tensor, prev_actions: tf.Tensor,
            old_prev_actor_outs: tf.Tensor, prev_actor_outs: tf.Tensor, n: int, batch_size: int) -> Tuple[tf.Tensor, tf.Tensor]:
        raise NotImplementedError()
    
    def actor_loss_critic_part(self, values_next_discounted: tf.Tensor, density_ratio: tf.Tensor):
        raise NotImplementedError()


class NoiseGaussianActorBase(AutocorrelatedActor):

    def __init__(self, observations_space: gym.Space, actions_space: gym.Space, layers: Optional[Tuple[int]],
                 beta_penalty: float, actions_bound: float, alpha: float = 0.8,
                 num_parallel_envs: int = 1, *args, **kwargs):
        super().__init__(observations_space, actions_space, layers, beta_penalty, actions_bound,
            required_prev_samples=1, *args, **kwargs)

        self._num_parallel_envs = num_parallel_envs
        self._alpha = alpha
        self._noise_dist = tfp.distributions.MultivariateNormalDiag(
            scale_diag=tf.exp(self.log_std),
        )
        
        self._last_noise = self._sample_noise()
        self._noise_init_mask = tf.ones(shape=(self._num_parallel_envs, 1))

    def _sample_noise(self):
        return self._noise_dist.sample(sample_shape=(self._num_parallel_envs, ))

    def update_ends(self, ends: np.array):
        self._noise_init_mask = tf.cast(ends, dtype=tf.float32)

    def prob(self, observations: tf.Tensor, actions: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        mean = self._forward(observations)
        dist = tfp.distributions.MultivariateNormalDiag(
            scale_diag=tf.exp(self.log_std)
        )

        return dist.prob(actions - mean), dist.log_prob(actions - mean)

    def act(self, observations: tf.Tensor, **kwargs) -> Tuple[tf.Tensor, tf.Tensor]:
        mean = self._forward(observations)

        noise_init = self._sample_noise()
        
        noise_cont = self._alpha * self._last_noise + tf.sqrt(1 - tf.square(self._alpha)) * noise_init
        noise = noise_init * self._noise_init_mask + noise_cont * (1 - self._noise_init_mask)
        self._last_noise = noise
        self._noise_init_mask = tf.zeros_like(self._noise_init_mask)

        actions = mean + noise
        return actions, mean

    def init_inverse_covariance_matrices(self, n: int, batch_size: int) -> Tuple[tf.Tensor, tf.Tensor]:
        lam0 = get_lambda_0(n - 1, self._alpha)
        lam1 = get_lambda_1(n - 1, self._alpha)
        lam0_c_prod = tf_utils.kronecker_prod(lam0, self._cov_matrix)
        lam1_c_prod = tf_utils.kronecker_prod(lam1, self._cov_matrix)
        lam0_c_prod_inv = tf.linalg.inv(lam0_c_prod)
        lam1_c_prod_inv = tf.linalg.inv(lam1_c_prod)

        rep_lam0_c_prod_inv = tf.repeat(tf.expand_dims(lam0_c_prod_inv, axis=0), batch_size, axis=0)
        rep_lam1_c_prod_inv = tf.repeat(tf.expand_dims(lam1_c_prod_inv, axis=0), batch_size, axis=0)

        return rep_lam0_c_prod_inv, rep_lam1_c_prod_inv
    
    
    @tf.function
    def estimate_noise_trajectory(self, prev_samples_len: tf.Tensor, prev_actions: tf.Tensor,
            prev_actor_outs: tf.Tensor, n: int, batch_size: int) -> Tuple[tf.Tensor, tf.Tensor]:
        alpha_coeffs = tf.pow(
            self._alpha,
            tf.range(1, n + 1, dtype=tf.float32)
        )
        noise_mask = tf.cast(prev_samples_len, tf.float32)
        mask = tf.expand_dims(tf.expand_dims(noise_mask, 1) * tf.expand_dims(alpha_coeffs, 0), 2)
        return (prev_actions - prev_actor_outs) * mask


class NoiseGaussianActorWithU(NoiseGaussianActorBase):
    @tf.function
    def calculate_q_noise(self, actions: tf.Tensor, actor_outs: tf.Tensor, lengths: tf.Tensor,
            prev_actions: tf.Tensor, prev_actor_outs: tf.Tensor, prev_samples_len: tf.Tensor, actor_outs_next: tf.Tensor):
        prev_u = tf.where(
            tf.expand_dims(prev_samples_len == 1, axis=1),
            actor_outs[:,0] + self._alpha * tf.squeeze(prev_actions - prev_actor_outs, axis=1),
            actions[:,0]
        )

        last_noise = tf.gather_nd(actions - actor_outs ,tf.maximum(0, tf.expand_dims(lengths, 1) - 1), batch_dims=1)
        last_u = actor_outs_next + self._alpha * last_noise
        return tf.stack([prev_u, last_u], axis=1)

    @tf.function
    def actor_loss_critic_part(self, values_next_discounted: tf.Tensor, density_ratio: tf.Tensor):
        return values_next_discounted * tf.stop_gradient(tf.expand_dims(density_ratio, 1))


AUTOCORRELATED_ACTORS = {
    'autocor+u': NoiseGaussianActorWithU
}


class ACERAC(BaseACERAgent):
    def __init__(self, observations_space: gym.Space, actions_space: gym.Space, actor_layers: Optional[Tuple[int]],
                 critic_layers: Optional[Tuple[int]], b: float = 3, n: int = 2, alpha: int = None,
                 td_clip: float = None, use_v: bool = False, noise_type='integrated',
                 *args, **kwargs):
        """Actor-Critic with Experience Replay and autocorrelated actions.

        Args:
            observations_space: observations' vectors Space
            actions_space: actions' vectors Space
            actor_layers: number of units in Actor's hidden layers
            critic_layers: number of units in Critic's hidden layers
            b: density ratio truncating coefficient
            n: update window size
            alpha: autocorrelation coefficient. If None, 1 - (1 / n) is set
        """

        self._n = n
        self._prev_n = 1
        self._noise_type = noise_type

        if alpha is None:
            self._alpha = 1 - (1 / n)
        else:
            self._alpha = alpha

        self._td_clip = td_clip
        self._use_q = not use_v

        super().__init__(observations_space, actions_space, actor_layers, critic_layers, *args, **kwargs)
        self._b = b

        self._lam0_c_prod_inv, self._lam1_c_prod_inv = self._actor.init_inverse_covariance_matrices(n, self._batch_size)

        self._data_loader = tf.data.Dataset.from_generator(
            self._experience_replay_generator,
            (tf.dtypes.float32, tf.dtypes.float32, self._actor.action_dtype, self._actor.action_dtype, tf.dtypes.float32,
             tf.dtypes.bool, tf.dtypes.int32, tf.dtypes.int32,
             tf.dtypes.float32, self._actor.action_dtype, self._actor.action_dtype)
        ).prefetch(2)
        
        self.logged_graph = False

    def _init_replay_buffer(self, memory_size: int):
        if type(self._actions_space) == gym.spaces.Discrete:
            actions_shape = (1,)
        else:
            actions_shape = self._actions_space.shape

        self._memory = MultiReplayBuffer(
            action_spec=BufferFieldSpec(shape=actions_shape, dtype=self._actor.action_dtype_np),
            obs_spec=BufferFieldSpec(shape=self._observations_space.shape, dtype=self._observations_space.dtype),
            max_size=memory_size,
            num_buffers=self._num_parallel_envs,
            buffer_class=PrevReplayBuffer(self._actor.required_prev_samples)
        )

    def _init_actor(self) -> BaseActor:
        if self._is_discrete:
            raise NotImplementedError
        else:
            return AUTOCORRELATED_ACTORS[self._noise_type](
                self._observations_space, self._actions_space, self._actor_layers,
                self._actor_beta_penalty, self._actions_bound, self._alpha, self._num_parallel_envs,
                self._std, self._tf_time_step
            )

    def _init_critic(self) -> Critic:
        return Critic(self._observations_space, self._critic_layers, self._tf_time_step, use_additional_input=self._use_q)

    def save_experience(self, steps: List[
        Tuple[Union[int, float, list], np.array, np.array, np.array, bool, bool]
    ]):
        super().save_experience(steps)

        self._actor.update_ends(np.array([[step[5]] for step in steps]))

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

    @tf.function #(experimental_relax_shapes=True)
    def _learn_from_experience_batch(self, obs: tf.Tensor, obs_next: tf.Tensor, actions: tf.Tensor,
                                     old_actor_outs: tf.Tensor, rewards: tf.Tensor, dones: tf.Tensor,
                                     lengths: tf.Tensor, prev_samples_len: tf.Tensor,
                                     prev_obs: tf.Tensor, prev_actions: tf.Tensor, old_prev_actor_outs: tf.Tensor):
        """Performs single learning step. Padded tensors are used here, final results
         are masked out with zeros"""

        obs = self._process_observations(obs)
        obs_next = self._process_observations(obs_next)
        prev_obs = self._process_observations(prev_obs)
        rewards = self._process_rewards(rewards)

        with tf.GradientTape(persistent=True) as tape:
            actor_outs = self._actor.act_deterministic(obs)
            actor_outs_next = self._actor.act_deterministic(obs_next)
            prev_actor_outs = self._actor.act_deterministic(prev_obs)

            if self._use_q:
                q_noise = self._actor.calculate_q_noise(actions, actor_outs, lengths, prev_actions, prev_actor_outs, prev_samples_len, actor_outs_next)
            else:
                q_noise = None
                
            c_invs = self._get_c_invs(prev_samples_len)

            critic_out = self._critic.value(tf.stack([obs[:,0], obs_next], axis=1), additional_input=q_noise)
            values_first, values_next = tf.split(tf.squeeze(critic_out), 2, axis=1)
            values_next = values_next * (1 - tf.cast(tf.expand_dims(dones, 1), tf.float32))

            # trajectories shorter than n mask
            mask = tf.expand_dims(tf.sequence_mask(lengths, maxlen=self._n, dtype=tf.float32), 2)

            eta = self._actor.estimate_noise_trajectory(prev_samples_len, prev_actions, prev_actor_outs, self._n, self._batch_size)
            mu = self._actor.estimate_noise_trajectory(prev_samples_len, prev_actions, old_prev_actor_outs, self._n, self._batch_size)
            
            actions_mu_diff_current = (actions - actor_outs - eta) * mask
            actions_mu_diff_old = (actions - old_actor_outs - mu) * mask

            density_ratio = self._compute_soft_truncated_density_ratio(
                actions_mu_diff_current, actions_mu_diff_old, c_invs
            )

            gamma_coeffs = tf.expand_dims(tf.pow(self._gamma, tf.range(1., self._n + 1)), axis=0)
            td_rewards = tf.reduce_sum(rewards * gamma_coeffs, axis=1, keepdims=True)

            values_next_discounted = tf.expand_dims(tf.pow(self._gamma, tf.cast(lengths, tf.float32)), 1) * values_next

            td = tf.squeeze((-values_first + td_rewards + values_next_discounted), axis=1)

            if self._td_clip is not None:
                td = tf.clip_by_value(td, -self._td_clip, self._td_clip)

            d = td * density_ratio

            c_mu = tf.matmul(tf.reshape(actions_mu_diff_current, (self._batch_size, -1, 1)), c_invs, transpose_a=True)
            c_mu_d = c_mu * tf.reshape(d, (-1, 1, 1))

            c_mu_mean = c_mu_d / tf.reshape(tf.cast(lengths, tf.float32), (-1, 1, 1))

            bounds_penalty = tf.scalar_mul(
                    self._actor.beta_penalty,
                    tf.square(tf.maximum(0.0, tf.abs(actor_outs) - self._actions_bound))
            )
            bounds_penalty = tf.squeeze(mask) * tf.reduce_sum(
                bounds_penalty,
                axis=2
            )

            bounds_penalty = tf.reduce_sum(bounds_penalty, axis=1) / tf.cast(lengths, tf.float32)
            actor_loss_critic_part = self._actor.actor_loss_critic_part(values_next_discounted, density_ratio)
            actor_loss = tf.matmul(tf.stop_gradient(c_mu_mean), tf.reshape(actor_outs, (self._batch_size, -1, 1)))
            actor_loss = -tf.reduce_mean(actor_loss) + tf.reduce_mean(bounds_penalty) -tf.reduce_mean(actor_loss_critic_part)

            d_mean = d / tf.cast(lengths, tf.float32)
            critic_loss = -tf.reduce_mean(tf.squeeze(values_first, axis=1) * tf.stop_gradient(d_mean))

        grads_actor = tape.gradient(actor_loss, self._actor.trainable_variables)
        if self._gradient_norm is not None:
            grads_actor = self._clip_gradient(grads_actor, self._actor_gradient_norm_median, 'actor')
        else:
            with tf.name_scope('actor'):
                tf.summary.scalar(f'gradient_norm', tf.linalg.global_norm(grads_actor), step=self._tf_time_step)
        grads_var_actor = zip(grads_actor, self._actor.trainable_variables)
        self._actor_optimizer.apply_gradients(grads_var_actor)

        with tf.name_scope('actor'):
            tf.summary.scalar(f'batch_actor_loss', actor_loss, step=self._tf_time_step)
            tf.summary.scalar(f'batch_bounds_penalty', tf.reduce_mean(bounds_penalty), step=self._tf_time_step)

        grads_critic = tape.gradient(critic_loss, self._critic.trainable_variables)
        if self._gradient_norm is not None:
            grads_critic = self._clip_gradient(grads_critic, self._critic_gradient_norm_median, 'critic')
        else:
            with tf.name_scope('critic'):
                tf.summary.scalar(f'gradient_norm', tf.linalg.global_norm(grads_critic), step=self._tf_time_step)
        grads_var_critic = zip(grads_critic, self._critic.trainable_variables)
        self._critic_optimizer.apply_gradients(grads_var_critic)

        with tf.name_scope('critic'):
            tf.summary.scalar(f'batch_critic_loss', critic_loss, step=self._tf_time_step)
            tf.summary.scalar(f'batch_values_first_mean', tf.reduce_mean(values_first), step=self._tf_time_step)

    def _compute_soft_truncated_density_ratio(
            self, actions_mu_diff_current: tf.Tensor, actions_mu_diff_old: tf.Tensor, c_invs: tf.Tensor
    ):
        actions_mu_diff_flatten = tf.reshape(actions_mu_diff_current, (self._batch_size, -1, 1))
        actions_mu_diff_old_flatten = tf.reshape(actions_mu_diff_old, (self._batch_size, -1, 1))
        exp_current = tf.matmul(
            tf.matmul(actions_mu_diff_flatten, c_invs, transpose_a=True),
            actions_mu_diff_flatten
        )
        exp_old = tf.matmul(
            tf.matmul(actions_mu_diff_old_flatten, c_invs, transpose_a=True),
            actions_mu_diff_old_flatten
        )
        density_ratio = tf.squeeze(tf.exp(-0.5 * exp_current + 0.5 * exp_old))
        density_ratio = tf.tanh(density_ratio / self._b) * self._b

        with tf.name_scope('acerac'):
            """tf.summary.scalar('mean_prob_current',
                tf.reduce_mean(tf.squeeze(tf.exp(-0.5 * exp_old)) / tf.sqrt(tf.linalg.det(c_invs) * (2 * np.pi) ** actions_mu_diff_current.shape[1])),
                step=self._tf_time_step)
            tf.summary.scalar('mean_prob_old',
                tf.reduce_mean(tf.squeeze(tf.exp(-0.5 * exp_old)) / tf.sqrt(tf.linalg.det(c_invs) * (2 * np.pi) ** actions_mu_diff_current.shape[1])),
                step=self._tf_time_step)
            """
            tf.summary.scalar('mean_density_ratio', tf.reduce_mean(density_ratio), step=self._tf_time_step)
            tf.summary.scalar('max_density_ratio', tf.reduce_max(density_ratio), step=self._tf_time_step)

        return density_ratio
        # return tf.minimum(density_ratio, self._b)
    
    def _get_c_invs(self, prev_samples_len: tf.Tensor) -> tf.Tensor:
        c_invs = tf.where(tf.reshape(prev_samples_len == self._actor.required_prev_samples, (-1, 1, 1)), self._lam1_c_prod_inv, self._lam0_c_prod_inv)
        return c_invs

    def _fetch_offline_batch(self) -> List[Tuple[Dict[str, Union[np.array, list]], int]]:
        batch = self._memory.get_vec(self._batches_per_env, self._n)
        return batch

    def _experience_replay_generator(self):
        """Generates trajectories batches. All tensors are padded with zeros to match self._n number of
        experience tuples in a single trajectory.
        Trajectories are returned in shape [batch, self._n, <obs/actions/etc shape>]
        """
        prev_n = self._actor.required_prev_samples
        while True:
            offline_batches, lens, prev_lens = self._fetch_offline_batch()
            
            lengths = lens
            obs = offline_batches['observations'][:,prev_n:]
            obs_next = offline_batches['next_observations'][np.arange(self._batch_size),prev_n + lens - 1]
            actions = offline_batches['actions'][:,prev_n:]
            rewards = offline_batches['rewards'][:,prev_n:]
            means = offline_batches['policies'][:,prev_n:]
            dones = offline_batches['dones'][np.arange(self._batch_size),prev_n + lens - 1]

            prev_samples_len = prev_lens

            # shift prev samples so that at 0 is the first actual sample
            i1, i2 = np.ogrid[:self._batch_size, :prev_n]
            i2 = i2 - prev_lens[:, np.newaxis]
            i2 = i2 % prev_n

            prev_obs = offline_batches['observations'][i1, i2]
            prev_actions = offline_batches['actions'][i1, i2]
            prev_means = offline_batches['policies'][i1, i2]

            yield (
                obs,
                obs_next,
                actions,
                means,
                rewards,
                dones,
                lengths,
                prev_samples_len,
                prev_obs,
                prev_actions,
                prev_means,
            )
