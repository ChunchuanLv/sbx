import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, List, Optional, Union

#import numpy as np
import jax.numpy as jnp
import torch as th
import gymnax.environments.spaces as spaces

from sbx.common.preprocessing import get_action_dim, get_obs_shape
from sbx.common.type_aliases import (
    DictReplayBufferSamples,
    DictRolloutBufferSamples,
    ReplayBufferSamples,
    RolloutBufferSamples,
    HierarchicalRolloutBufferSamples,
    EmpowermentHierarchicalRolloutBufferSamples
)
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.vec_env import VecNormalize

try:
    # Check memory used by replay buffer when possible
    import psutil
except ImportError:
    psutil = None
import jax

class BaseBuffer(ABC):
    """
    Base class that represent a buffer (rollout or replay)

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
        to which the values will be converted
    :param n_envs: Number of parallel environments
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
    ):
        super().__init__()
        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.obs_shape = get_obs_shape(observation_space)

        self.action_dim = get_action_dim(action_space)
        self.pos = 0
        self.full = False
        self.device = get_device(device)
        self.n_envs = n_envs

    @staticmethod
    def swap_and_flatten(arr: Union[jnp.ndarray,List[jnp.ndarray]]) -> jnp.ndarray:
        """
        Swap and then flatten axes 0 (buffer_size) and 1 (n_envs)
        to convert shape from [n_steps, n_envs, ...] (when ... is the shape of the features)
        to [n_steps * n_envs, ...] (which maintain the order)

        :param arr:
        :return:
        """
        if isinstance(arr, jnp.ndarray):
            shape = arr.shape
            if len(shape) < 3:
                shape = (*shape, 1)

            return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])
        else:
            arr = jnp.concatenate(arr, axis=0)

    @staticmethod
    def flatten(arr: Union[jnp.ndarray,List[jnp.ndarray]]) -> jnp.ndarray:
        """
        Swap and then flatten axes 0 (buffer_size) and 1 (n_envs)
        to convert shape from [n_steps, n_envs, ...] (when ... is the shape of the features)
        to [n_steps * n_envs, ...] (which maintain the order)

        :param arr:
        :return:
        """
        if isinstance(arr, jnp.ndarray):
            shape = arr.shape
            if len(shape) < 3:
                shape = (*shape, 1)

            return arr.reshape(shape[0] * shape[1], *shape[2:])
        else:
            return jnp.concatenate(arr, axis=0)
    def size(self) -> int:
        """
        :return: The current size of the buffer
        """
        if self.full:
            return self.buffer_size
        return self.pos

    def add(self, *args, **kwargs) -> None:
        """
        Add elements to the buffer.
        """
        raise NotImplementedError()

    def extend(self, *args, **kwargs) -> None:
        """
        Add a new batch of transitions to the buffer
        """
        # Do a for loop along the batch axis
        for data in zip(*args):
            self.add(*data)

    def reset(self) -> None:
        """
        Reset the buffer.
        """
        self.pos = 0
        self.full = False

    def sample(self, key, batch_size: int, env: Optional[VecNormalize] = None):
        """
        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = jnp.random.randint(key,(batch_size),0, upper_bound)
        return self._get_samples(batch_inds, env=env)

    @abstractmethod
    def _get_samples(
        self, batch_inds: jnp.ndarray, env: Optional[VecNormalize] = None
    ) -> Union[ReplayBufferSamples, RolloutBufferSamples]:
        """
        :param batch_inds:
        :param env:
        :return:
        """
        raise NotImplementedError()


    @staticmethod
    def _normalize_obs(
        obs: Union[jnp.ndarray, Dict[str, jnp.ndarray]],
        env: Optional[VecNormalize] = None,
    ) -> Union[jnp.ndarray, Dict[str, jnp.ndarray]]:
        if env is not None:
            return env.normalize_obs(obs)
        return obs

    @staticmethod
    def _normalize_reward(reward: jnp.ndarray, env: Optional[VecNormalize] = None) -> jnp.ndarray:
        if env is not None:
            return env.normalize_reward(reward).astype(jnp.float32)
        return reward


class ReplayBuffer(BaseBuffer):
    """
    Replay buffer used in off-policy algorithms like SAC/TD3.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Enable a memory efficient variant
        of the replay buffer which reduces by almost a factor two the memory used,
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
        and https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274
        Cannot be used in combination with handle_timeout_termination.
    :param handle_timeout_termination: Handle timeout termination (due to timelimit)
        separately and treat the task as infinite horizon task.
        https://github.com/DLR-RM/stable-baselines3/issues/284
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = False,
    ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)

        # Adjust buffer size
        self.buffer_size = max(buffer_size // n_envs, 1)

        # Check that the replay buffer can fit into the memory
        if psutil is not None:
            mem_available = psutil.virtual_memory().available

        # there is a bug if both optimize_memory_usage and handle_timeout_termination are true
        # see https://github.com/DLR-RM/stable-baselines3/issues/934
        if optimize_memory_usage and handle_timeout_termination:
            raise ValueError(
                "ReplayBuffer does not support optimize_memory_usage = True "
                "and handle_timeout_termination = True simultaneously."
            )
        self.optimize_memory_usage = optimize_memory_usage

        self.observations = jnp.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=observation_space.dtype)

        if optimize_memory_usage:
            # `observations` contains also the next observation
            self.next_observations = None
        else:
            self.next_observations = jnp.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=observation_space.dtype)

        self.actions = jnp.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=action_space.dtype)

        self.rewards = jnp.zeros((self.buffer_size, self.n_envs), dtype=jnp.float32)
        self.dones = jnp.zeros((self.buffer_size, self.n_envs), dtype=jnp.float32)
        # Handle timeouts termination properly if needed
        # see https://github.com/DLR-RM/stable-baselines3/issues/284
        self.handle_timeout_termination = handle_timeout_termination
        self.timeouts = jnp.zeros((self.buffer_size, self.n_envs), dtype=jnp.float32)

        if psutil is not None:
            total_memory_usage = self.observations.nbytes + self.actions.nbytes + self.rewards.nbytes + self.dones.nbytes

            if self.next_observations is not None:
                total_memory_usage += self.next_observations.nbytes

            if total_memory_usage > mem_available:
                # Convert to GB
                total_memory_usage /= 1e9
                mem_available /= 1e9
                warnings.warn(
                    "This system does not have apparently enough memory to store the complete "
                    f"replay buffer {total_memory_usage:.2f}GB > {mem_available:.2f}GB"
                )

    def add(
        self,
        obs: jnp.ndarray,
        next_obs: jnp.ndarray,
        action: jnp.ndarray,
        reward: jnp.ndarray,
        done: jnp.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))
            next_obs = next_obs.reshape((self.n_envs, *self.obs_shape))

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        # Copy to avoid modification by reference
        self.observations[self.pos] = jnp.array(obs).copy()

        if self.optimize_memory_usage:
            self.observations[(self.pos + 1) % self.buffer_size] = jnp.array(next_obs).copy()
        else:
            self.next_observations[self.pos] = jnp.array(next_obs).copy()

        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done

        if self.handle_timeout_termination:
            self.timeouts[self.pos] = jnp.array([info.get("TimeLimit.truncated", False) for info in infos])

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, key,batch_size: int, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        """
        Sample elements from the replay buffer.
        Custom sampling when using memory efficient variant,
        as we should not sample the element with index `self.pos`
        See https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274

        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        if not self.optimize_memory_usage:
            return super().sample(key,batch_size=batch_size, env=env)
        # Do not sample the element with index `self.pos` as the transitions is invalid
        # (we use only one array to store `obs` and `next_obs`)
        if self.full:
            batch_inds = (jnp.random.randint(key,(batch_size),1, self.buffer_size) + self.pos) % self.buffer_size
        else:
            batch_inds = jnp.random.randint(key,(batch_size),0, self.pos)
        return self._get_samples(batch_inds, env=env)

    def _get_samples(self, batch_inds: jnp.ndarray, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        # Sample randomly the env idx
        env_indices = jnp.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :], env)
        else:
            next_obs = self._normalize_obs(self.next_observations[batch_inds, env_indices, :], env)

        data = (
            self._normalize_obs(self.observations[batch_inds, env_indices, :], env),
            self.actions[batch_inds, env_indices, :],
            next_obs,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
            self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env),
        )
        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))


class EmpowermentHierarchicalRolloutBuffer(BaseBuffer):
    """
    Rollout buffer used in on-policy algorithms like A2C/PPO.
    It corresponds to ``buffer_size`` transitions collected
    using the current policy.
    This experience will be discarded after the policy update.
    In order to use PPO objective, we also store the current value of each state
    and the log probability of each taken action.

    The term rollout here refers to the model-free notion and should not
    be used with the concept of rollout used in model-based RL or planning.
    Hence, it is only involved in policy and value function training but not action selection.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to classic advantage when set to 1.
    :param gamma: Discount factor
    :param n_envs: Number of parallel environments
    """

    observations: List[jnp.ndarray]
    actions:  List[jnp.ndarray]
    rewards:  List[jnp.ndarray]
    advantages:  List[jnp.ndarray]
    returns:  List[jnp.ndarray]
    episode_starts:  List[jnp.ndarray]
    log_probs: List[ jnp.ndarray]
    values:  List[jnp.ndarray]

    options:  List[jnp.ndarray]
    option_starts:  List[jnp.ndarray]
    option_start_advantages:  List[jnp.ndarray]
    option_values:  List[jnp.ndarray]
    option_advantages:  List[jnp.ndarray]
    total_values:  List[jnp.ndarray]

    variational_posterior :  List[jnp.ndarray]
    entropy:  List[jnp.ndarray]
    option_start_time :  List[jnp.ndarray]
    control_values: List[jnp.ndarray]
    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        n_options: int,
        device: Union[th.device, str] = "auto",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
    ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)
        self.n_options = n_options
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.generator_ready = False
        self.reset()

    def reset(self) -> None:
        super().reset()
        self.observations = [jnp.zeros(( self.n_envs, *self.obs_shape), dtype=jnp.float32) ] * self.buffer_size
        self.actions = [jnp.zeros(( self.n_envs, self.action_dim), dtype=jnp.float32)] * self.buffer_size
        self.rewards = [jnp.zeros(( self.n_envs), dtype=jnp.float32)] * self.buffer_size
        self.returns = [jnp.zeros(( self.n_envs), dtype=jnp.float32)] * self.buffer_size
        self.episode_starts = [jnp.zeros(( self.n_envs), dtype=bool)] * self.buffer_size
        self.values = [jnp.zeros((self.n_envs), dtype=jnp.float32)] * self.buffer_size
        self.log_probs = [jnp.zeros(( self.n_envs), dtype=jnp.float32)] * self.buffer_size
        self.advantages = [jnp.zeros(( self.n_envs), dtype=jnp.float32)] * self.buffer_size

        self.option_returns = [jnp.zeros(( self.n_envs), dtype=jnp.float32)] * self.buffer_size
        self.option_advantages = [jnp.zeros(( self.n_envs), dtype=jnp.float32)] * self.buffer_size
        self.option_start_advantages = [jnp.zeros(( self.n_envs), dtype=jnp.float32)] * self.buffer_size

        self.options = [jnp.zeros(( self.n_envs, self.n_options), dtype=int)] * self.buffer_size
        self.option_starts = [jnp.zeros(( self.n_envs), dtype=bool)] * self.buffer_size
        self.option_start_log_probs = [jnp.zeros(( self.n_envs), dtype=jnp.float32)] * self.buffer_size
        self.option_log_probs = [jnp.zeros(( self.n_envs), dtype=jnp.float32)] * self.buffer_size
        self.option_values = [jnp.zeros(( self.n_envs), dtype=jnp.float32)] * self.buffer_size
        self.total_values = [jnp.zeros(( self.n_envs), dtype=jnp.float32)] * self.buffer_size


        self.variational_posterior = [jnp.zeros(( self.n_envs), dtype=jnp.float32)] * self.buffer_size
        self.control_values = [jnp.zeros(( self.n_envs), dtype=jnp.float32)] * self.buffer_size
        self.entropy = [jnp.zeros(( self.n_envs), dtype=jnp.float32)] * self.buffer_size
        self.option_start_time = [jnp.zeros(( self.n_envs), dtype=jnp.int32)] * self.buffer_size
        self.generator_ready = False

    def compute_returns_and_advantage(self, last_values: jnp.ndarray,last_option_values:jnp.ndarray,
                                      last_control_values:jnp.ndarray, last_entropy,last_variational_posterior,
                                      last_option_start_time:jnp.ndarray,
                                      last_option_start_log_prob:jnp.ndarray, dones: jnp.ndarray) -> None:
        """
        Post-processing step: compute the lambda-return (TD(lambda) estimate)
        and GAE(lambda) advantage.

        Uses Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)
        to compute the advantage. To obtain Monte-Carlo advantage estimate (A(s) = R - V(S))
        where R is the sum of discounted reward with value bootstrap
        (because we don't always have full episode), set ``gae_lambda=1.0`` during initialization.

        The TD(lambda) estimator has also two special cases:
        - TD(1) is Monte-Carlo estimate (sum of discounted rewards)
        - TD(0) is one-step estimate with bootstrapping (r_t + gamma * v(s_{t+1}))

        For more information, see discussion in https://github.com/DLR-RM/stable-baselines3/pull/375.

        :param last_values: state value estimation for the last step (one for each env)
        :param dones: if the last step was a terminal step (one bool for each env).
        """
        # Convert to numpy
        last_values = last_values.flatten()
        last_option_values = last_option_values.flatten()
        last_option_starts_probability = jnp.exp(last_option_start_log_prob.flatten())

        last_gae_lam = 0

        last_value_adjusted = (last_values + self.gamma**(last_option_start_time-self.buffer_size) * last_variational_posterior)
        last_option_value_adjusted = last_option_values- last_entropy-last_control_values * ( 1- self.gamma**(last_option_start_time-self.buffer_size) )
        self.total_values[self.pos] =last_value_adjusted * last_option_starts_probability + last_option_value_adjusted* (1-last_option_starts_probability)

        last_total_value = last_values * last_option_starts_probability + last_option_values * (1-last_option_starts_probability)
        gae_lambda_multiplier = (1 - self.gae_lambda ) / (1 - self.gae_lambda ** self.buffer_size)
        gamme_lambda = self.gamma * self.gae_lambda
        generalized_reward = 0
        generalized_value = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones
                next_total_value = last_total_value 
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_total_value = self.total_values[step + 1]

            generalized_value = generalized_value * gamme_lambda + self.gamma * next_total_value * next_non_terminal / gae_lambda_multiplier
            generalized_reward = generalized_reward * gamme_lambda + self.rewards[step] / gae_lambda_multiplier
            generalized_returns = gae_lambda_multiplier * (generalized_reward + generalized_value)
            
            self.option_advantages[step] = generalized_returns - self.values[step]
            self.option_start_advantages[step] = generalized_returns - self.values[step] 
            self.advantages[step] = generalized_returns - self.option_values[step]
            
        # TD(lambda) estimator, see Github PR #375 or "Telescoping in TD(lambda)"
        # in David Silver Lecture 4: https://www.youtube.com/watch?v=PnHCvfgC_ZA
            self.returns[step] = generalized_returns

    def add(
        self,
        obs: jnp.ndarray,
        action: jnp.ndarray,
        reward: jnp.ndarray,
        episode_start: jnp.ndarray,
        value: jnp.ndarray,
        option:jnp.ndarray,
        option_start:jnp.ndarray,
        option_value:jnp.ndarray,
        log_prob: jnp.ndarray,
        option_log_prob: jnp.ndarray,
        option_start_log_prob: jnp.ndarray,
        variational_posterior: jnp.ndarray,
        control_values: jnp.ndarray,
        entropy: jnp.ndarray,
        option_start_time:jnp.ndarray,
    ) -> None:
        """
        :param obs: Observation
        :param action: Action
        :param reward:
        :param episode_start: Start of episode signal.
        :param value: estimated value of the current state
            following the current policy.
        :param log_prob: log probability of the action
            following the current policy.
        :param variational_posterior: variational_posterior of the current state
        """
        if len(log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = log_prob.reshape(-1, 1)

        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))
        option_value = option_value.flatten()
        value = value.flatten()
        variational_posterior = variational_posterior * (1-episode_start)
        self.observations[self.pos] = obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward+option_start* (entropy+self.gamma**(option_start_time-self.pos) * variational_posterior)
        self.episode_starts[self.pos] = episode_start
        self.values[self.pos] = value
        self.options[self.pos] = option
        self.option_values[self.pos] = option_value
        self.option_starts[self.pos] = option_start
        option_start_prob = jnp.exp(option_start_log_prob.flatten())
        value_adjusted = (value + self.gamma**(option_start_time-self.pos) * variational_posterior)
        option_value_adjusted = option_value- entropy-control_values * ( 1- self.gamma**(option_start_time-self.pos) )
        self.total_values[self.pos] =value_adjusted * option_start_prob + option_value_adjusted* (1-option_start_prob)
        self.log_probs[self.pos] = log_prob
        self.option_start_log_probs[self.pos] = option_start_log_prob
        self.option_log_probs[self.pos] = option_log_prob #* option_start_prob + option_start_log_prob * (1-option_start_prob)
        self.variational_posterior[self.pos] = variational_posterior
        self.entropy[self.pos] = entropy
        self.option_start_time[self.pos] = option_start_time
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def get(self, random_key, batch_size: Optional[int] = None) -> Generator[RolloutBufferSamples, None, None]:
        assert self.full, ""
        indices = jax.random.permutation(random_key,self.buffer_size * self.n_envs)
        # Prepare the data
        _tensor_names = [
            "observations",
            "actions",
            "options",
            "option_starts",
            "episode_starts",
            "log_probs",
            "option_log_probs",
            "option_start_log_probs",
            "advantages",
            "option_advantages",
            "option_start_advantages",
            "returns",
            "variational_posterior"
            "control_values",
            "option_start_time",
            "entropy"
        ]
        samples = {}
        for tensor in _tensor_names:
            samples[tensor] = self.flatten(self.__dict__[tensor])
        samples["last_options"] = self.flatten( [self.__dict__["options"][-1]] + self.__dict__["options"][:-1] )
        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size],samples)
            start_idx += batch_size

    def _get_samples(
        self,
        batch_inds: jnp.ndarray,
        samples: Dict[str, jnp.ndarray],
        env: Optional[VecNormalize] = None,
    ) -> EmpowermentHierarchicalRolloutBufferSamples:  # type: ignore[signature-mismatch] #FIXME
        data = (
            samples["observations"][batch_inds],
            samples["actions"][batch_inds],
            samples["options"][batch_inds],
            samples["option_starts"][batch_inds],
            samples["episode_starts"][batch_inds],
            samples["last_options"][batch_inds],
            samples["log_probs"][batch_inds].flatten(),
            samples["option_log_probs"][batch_inds].flatten(),
            samples["option_start_log_probs"][batch_inds].flatten(),
            samples["advantages"][batch_inds].flatten(),
            samples["option_advantages"][batch_inds].flatten(),
            samples["option_start_advantages"][batch_inds].flatten(),
            samples["returns"][batch_inds].flatten(),
            samples["variational_posterior"][batch_inds].flatten(),
            samples["control_values"][batch_inds].flatten(),
            samples["option_start_time"][batch_inds].flatten(),
            samples["entropy"][batch_inds].flatten(),
        )
     #   print ("data.options",data.options)
   #     print ("data.last_options",data.last_options)
        return EmpowermentHierarchicalRolloutBufferSamples(*data)

    class HierarchicalRolloutBuffer(BaseBuffer):
        """
        Rollout buffer used in on-policy algorithms like A2C/PPO.
        It corresponds to ``buffer_size`` transitions collected
        using the current policy.
        This experience will be discarded after the policy update.
        In order to use PPO objective, we also store the current value of each state
        and the log probability of each taken action.

        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.
        Hence, it is only involved in policy and value function training but not action selection.

        :param buffer_size: Max number of element in the buffer
        :param observation_space: Observation space
        :param action_space: Action space
        :param device: PyTorch device
        :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
            Equivalent to classic advantage when set to 1.
        :param gamma: Discount factor
        :param n_envs: Number of parallel environments
        """

        observations: List[jnp.ndarray]
        actions: List[jnp.ndarray]
        rewards: List[jnp.ndarray]
        advantages: List[jnp.ndarray]
        returns: List[jnp.ndarray]
        episode_starts: List[jnp.ndarray]
        log_probs: List[jnp.ndarray]
        values: List[jnp.ndarray]

        options: List[jnp.ndarray]
        option_starts: List[jnp.ndarray]
        option_start_advantages: List[jnp.ndarray]
        option_values: List[jnp.ndarray]
        option_advantages: List[jnp.ndarray]
        total_values: List[jnp.ndarray]

        def __init__(
                self,
                buffer_size: int,
                observation_space: spaces.Space,
                action_space: spaces.Space,
                n_options: int,
                device: Union[th.device, str] = "auto",
                gae_lambda: float = 1,
                gamma: float = 0.99,
                n_envs: int = 1,
        ):
            super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)
            self.n_options = n_options
            self.gae_lambda = gae_lambda
            self.gamma = gamma
            self.generator_ready = False
            self.reset()

        def reset(self) -> None:
            super().reset()
            self.observations = [jnp.zeros((self.n_envs, *self.obs_shape), dtype=jnp.float32)] * self.buffer_size
            self.actions = [jnp.zeros((self.n_envs, self.action_dim), dtype=jnp.float32)] * self.buffer_size
            self.rewards = [jnp.zeros((self.n_envs), dtype=jnp.float32)] * self.buffer_size
            self.returns = [jnp.zeros((self.n_envs), dtype=jnp.float32)] * self.buffer_size
            self.episode_starts = [jnp.zeros((self.n_envs), dtype=bool)] * self.buffer_size
            self.values = [jnp.zeros((self.n_envs), dtype=jnp.float32)] * self.buffer_size
            self.log_probs = [jnp.zeros((self.n_envs), dtype=jnp.float32)] * self.buffer_size
            self.advantages = [jnp.zeros((self.n_envs), dtype=jnp.float32)] * self.buffer_size

            self.option_returns = [jnp.zeros((self.n_envs), dtype=jnp.float32)] * self.buffer_size
            self.option_advantages = [jnp.zeros((self.n_envs), dtype=jnp.float32)] * self.buffer_size
            self.option_start_advantages = [jnp.zeros((self.n_envs), dtype=jnp.float32)] * self.buffer_size

            self.options = [jnp.zeros((self.n_envs, self.n_options), dtype=int)] * self.buffer_size
            self.option_starts = [jnp.zeros((self.n_envs), dtype=bool)] * self.buffer_size
            self.option_start_log_probs = [jnp.zeros((self.n_envs), dtype=jnp.float32)] * self.buffer_size
            self.option_log_probs = [jnp.zeros((self.n_envs), dtype=jnp.float32)] * self.buffer_size
            self.option_values = [jnp.zeros((self.n_envs), dtype=jnp.float32)] * self.buffer_size
            self.total_values = [jnp.zeros((self.n_envs), dtype=jnp.float32)] * self.buffer_size
            self.generator_ready = False

        def compute_returns_and_advantage(self, last_values: jnp.ndarray, last_option_values: jnp.ndarray,
                                          last_option_start_log_prob: jnp.ndarray, dones: jnp.ndarray) -> None:
            """
            Post-processing step: compute the lambda-return (TD(lambda) estimate)
            and GAE(lambda) advantage.

            Uses Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)
            to compute the advantage. To obtain Monte-Carlo advantage estimate (A(s) = R - V(S))
            where R is the sum of discounted reward with value bootstrap
            (because we don't always have full episode), set ``gae_lambda=1.0`` during initialization.

            The TD(lambda) estimator has also two special cases:
            - TD(1) is Monte-Carlo estimate (sum of discounted rewards)
            - TD(0) is one-step estimate with bootstrapping (r_t + gamma * v(s_{t+1}))

            For more information, see discussion in https://github.com/DLR-RM/stable-baselines3/pull/375.

            :param last_values: state value estimation for the last step (one for each env)
            :param dones: if the last step was a terminal step (one bool for each env).
            """
            # Convert to numpy
            last_values = last_values.flatten()
            last_option_values = last_option_values.flatten()
            last_option_starts_probability = jnp.exp(last_option_start_log_prob.flatten())

            last_gae_lam = 0

            last_total_value = last_values * last_option_starts_probability + last_option_values * (
                        1 - last_option_starts_probability)
            gae_lambda_multiplier = (1 - self.gae_lambda) / (1 - self.gae_lambda ** self.buffer_size)
            gamme_lambda = self.gamma * self.gae_lambda
            generalized_reward = 0
            generalized_value = 0
            for step in reversed(range(self.buffer_size)):
                if step == self.buffer_size - 1:
                    next_non_terminal = 1.0 - dones
                    next_total_value = last_total_value
                else:
                    next_non_terminal = 1.0 - self.episode_starts[step + 1]
                    next_total_value = self.total_values[step + 1]

                generalized_value = generalized_value * gamme_lambda + self.gamma * next_total_value * next_non_terminal / gae_lambda_multiplier
                generalized_reward = generalized_reward * gamme_lambda + self.rewards[step] / gae_lambda_multiplier
                generalized_returns = gae_lambda_multiplier * (generalized_reward + generalized_value)

                self.option_advantages[step] = generalized_returns - self.values[step]
                self.option_start_advantages[step] = generalized_returns - self.values[step]
                self.advantages[step] = generalized_returns - self.option_values[step]

                # TD(lambda) estimator, see Github PR #375 or "Telescoping in TD(lambda)"
                # in David Silver Lecture 4: https://www.youtube.com/watch?v=PnHCvfgC_ZA
                self.returns[step] = generalized_returns

        def add(
                self,
                obs: jnp.ndarray,
                action: jnp.ndarray,
                reward: jnp.ndarray,
                episode_start: jnp.ndarray,
                value: jnp.ndarray,
                option: jnp.ndarray,
                option_start: jnp.ndarray,
                option_value: jnp.ndarray,
                log_prob: jnp.ndarray,
                option_log_prob: jnp.ndarray,
                option_start_log_prob: jnp.ndarray,
        ) -> None:
            """
            :param obs: Observation
            :param action: Action
            :param reward:
            :param episode_start: Start of episode signal.
            :param value: estimated value of the current state
                following the current policy.
            :param log_prob: log probability of the action
                following the current policy.
            """
            if len(log_prob.shape) == 0:
                # Reshape 0-d tensor to avoid error
                log_prob = log_prob.reshape(-1, 1)

            # Reshape needed when using multiple envs with discrete observations
            # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
            if isinstance(self.observation_space, spaces.Discrete):
                obs = obs.reshape((self.n_envs, *self.obs_shape))

            # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
            action = action.reshape((self.n_envs, self.action_dim))

            self.observations[self.pos] = obs
            self.actions[self.pos] = action
            self.rewards[self.pos] = reward
            self.episode_starts[self.pos] = episode_start
            self.values[self.pos] = value.flatten()
            self.options[self.pos] = option
            self.option_values[self.pos] = option_value.flatten()
            self.option_starts[self.pos] = option_start
            option_start_prob = jnp.exp(option_start_log_prob.flatten())
            self.total_values[self.pos] = value.flatten() * option_start_prob + option_value.flatten() * (
                        1 - option_start_prob)
            self.log_probs[self.pos] = log_prob
            self.option_start_log_probs[self.pos] = option_start_log_prob
            self.option_log_probs[
                self.pos] = option_log_prob  # * option_start_prob + option_start_log_prob * (1-option_start_prob)
            self.pos += 1
            if self.pos == self.buffer_size:
                self.full = True

        def get(self, random_key, batch_size: Optional[int] = None) -> Generator[RolloutBufferSamples, None, None]:
            assert self.full, ""
            indices = jax.random.permutation(random_key, self.buffer_size * self.n_envs)
            # Prepare the data
            _tensor_names = [
                "observations",
                "actions",
                "options",
                "option_starts",
                "episode_starts",
                "log_probs",
                "option_log_probs",
                "option_start_log_probs",
                "advantages",
                "option_advantages",
                "option_start_advantages",
                "returns",
            ]
            samples = {}
            for tensor in _tensor_names:
                samples[tensor] = self.flatten(self.__dict__[tensor])
            samples["last_options"] = self.flatten([self.__dict__["options"][-1]] + self.__dict__["options"][:-1])
            # Return everything, don't create minibatches
            if batch_size is None:
                batch_size = self.buffer_size * self.n_envs

            start_idx = 0
            while start_idx < self.buffer_size * self.n_envs:
                yield self._get_samples(indices[start_idx: start_idx + batch_size], samples)
                start_idx += batch_size

        def _get_samples(
                self,
                batch_inds: jnp.ndarray,
                samples: Dict[str, jnp.ndarray],
                env: Optional[VecNormalize] = None,
        ) -> HierarchicalRolloutBufferSamples:  # type: ignore[signature-mismatch] #FIXME
            data = (
                samples["observations"][batch_inds],
                samples["actions"][batch_inds],
                samples["options"][batch_inds],
                samples["option_starts"][batch_inds],
                samples["episode_starts"][batch_inds],
                samples["last_options"][batch_inds],
                samples["log_probs"][batch_inds].flatten(),
                samples["option_log_probs"][batch_inds].flatten(),
                samples["option_start_log_probs"][batch_inds].flatten(),
                samples["advantages"][batch_inds].flatten(),
                samples["option_advantages"][batch_inds].flatten(),
                samples["option_start_advantages"][batch_inds].flatten(),
                samples["returns"][batch_inds].flatten(),
            )
            data = HierarchicalRolloutBufferSamples(*data)
            #   print ("data.options",data.options)
            #     print ("data.last_options",data.last_options)
            return HierarchicalRolloutBufferSamples(*data)
class RolloutBuffer(BaseBuffer):
    """
    Rollout buffer used in on-policy algorithms like A2C/PPO.
    It corresponds to ``buffer_size`` transitions collected
    using the current policy.
    This experience will be discarded after the policy update.
    In order to use PPO objective, we also store the current value of each state
    and the log probability of each taken action.

    The term rollout here refers to the model-free notion and should not
    be used with the concept of rollout used in model-based RL or planning.
    Hence, it is only involved in policy and value function training but not action selection.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to classic advantage when set to 1.
    :param gamma: Discount factor
    :param n_envs: Number of parallel environments
    """

    observations: List[jnp.ndarray]
    actions:  List[jnp.ndarray]
    rewards:  List[jnp.ndarray]
    advantages:  List[jnp.ndarray]
    returns:  List[jnp.ndarray]
    episode_starts:  List[jnp.ndarray]
    log_probs: List[ jnp.ndarray]
    values:  List[jnp.ndarray]

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
    ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.generator_ready = False
        self.reset()

    def reset(self) -> None:
        super().reset()
        self.observations = [jnp.zeros(( self.n_envs, *self.obs_shape), dtype=jnp.float32) ] * self.buffer_size
        self.actions = [jnp.zeros(( self.n_envs, self.action_dim), dtype=jnp.float32)] * self.buffer_size
        self.rewards = [jnp.zeros(( self.n_envs), dtype=jnp.float32)] * self.buffer_size
        self.returns = [jnp.zeros(( self.n_envs), dtype=jnp.float32)] * self.buffer_size
        self.episode_starts = [jnp.zeros(( self.n_envs), dtype=jnp.float32)] * self.buffer_size
        self.values = [jnp.zeros((self.n_envs), dtype=jnp.float32)] * self.buffer_size
        self.log_probs = [jnp.zeros(( self.n_envs), dtype=jnp.float32)] * self.buffer_size
        self.advantages = [jnp.zeros(( self.n_envs), dtype=jnp.float32)] * self.buffer_size
        self.generator_ready = False

    def compute_returns_and_advantage(self, last_values: jnp.ndarray, dones: jnp.ndarray) -> None:
        """
        Post-processing step: compute the lambda-return (TD(lambda) estimate)
        and GAE(lambda) advantage.

        Uses Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)
        to compute the advantage. To obtain Monte-Carlo advantage estimate (A(s) = R - V(S))
        where R is the sum of discounted reward with value bootstrap
        (because we don't always have full episode), set ``gae_lambda=1.0`` during initialization.

        The TD(lambda) estimator has also two special cases:
        - TD(1) is Monte-Carlo estimate (sum of discounted rewards)
        - TD(0) is one-step estimate with bootstrapping (r_t + gamma * v(s_{t+1}))

        For more information, see discussion in https://github.com/DLR-RM/stable-baselines3/pull/375.

        :param last_values: state value estimation for the last step (one for each env)
        :param dones: if the last step was a terminal step (one bool for each env).
        """
        # Convert to numpy
        last_values = last_values.flatten()

        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]
            delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam
        # TD(lambda) estimator, see Github PR #375 or "Telescoping in TD(lambda)"
        # in David Silver Lecture 4: https://www.youtube.com/watch?v=PnHCvfgC_ZA
            self.returns[step] = self.advantages[step] + self.values[step]

    def add(
        self,
        obs: jnp.ndarray,
        action: jnp.ndarray,
        reward: jnp.ndarray,
        episode_start: jnp.ndarray,
        value: jnp.ndarray,
        log_prob: jnp.ndarray,
    ) -> None:
        """
        :param obs: Observation
        :param action: Action
        :param reward:
        :param episode_start: Start of episode signal.
        :param value: estimated value of the current state
            following the current policy.
        :param log_prob: log probability of the action
            following the current policy.
        """
        if len(log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = log_prob.reshape(-1, 1)

        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        self.observations[self.pos] = obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.episode_starts[self.pos] = episode_start
        self.values[self.pos] = value.flatten()
        self.log_probs[self.pos] = log_prob
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def get(self, random_key, batch_size: Optional[int] = None) -> Generator[RolloutBufferSamples, None, None]:
        assert self.full, ""
        indices = jax.random.permutation(random_key,self.buffer_size * self.n_envs)
        # Prepare the data
        _tensor_names = [
            "observations",
            "actions",
            "values",
            "log_probs",
            "advantages",
            "returns",
        ]
        samples = {}
        for tensor in _tensor_names:
            samples[tensor] = self.flatten(self.__dict__[tensor])

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size],samples)
            start_idx += batch_size

    def _get_samples(
        self,
        batch_inds: jnp.ndarray,
        samples: Dict[str, jnp.ndarray],
        env: Optional[VecNormalize] = None,
    ) -> RolloutBufferSamples:  # type: ignore[signature-mismatch] #FIXME
        data = (
            samples["observations"][batch_inds],
            samples["actions"][batch_inds],
            samples["values"][batch_inds].flatten(),
            samples["log_probs"][batch_inds].flatten(),
            samples["advantages"][batch_inds].flatten(),
            samples["returns"][batch_inds].flatten(),
        )
        return RolloutBufferSamples(*data)


class DictReplayBuffer(ReplayBuffer):
    """
    Dict Replay buffer used in off-policy algorithms like SAC/TD3.
    Extends the ReplayBuffer to use dictionary observations

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Enable a memory efficient variant
        Disabled for now (see https://github.com/DLR-RM/stable-baselines3/pull/243#discussion_r531535702)
    :param handle_timeout_termination: Handle timeout termination (due to timelimit)
        separately and treat the task as infinite horizon task.
        https://github.com/DLR-RM/stable-baselines3/issues/284
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
    ):
        super(ReplayBuffer, self).__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)

        assert isinstance(self.obs_shape, dict), "DictReplayBuffer must be used with Dict obs space only"
        self.buffer_size = max(buffer_size // n_envs, 1)

        # Check that the replay buffer can fit into the memory
        if psutil is not None:
            mem_available = psutil.virtual_memory().available

        assert optimize_memory_usage is False, "DictReplayBuffer does not support optimize_memory_usage"
        # disabling as this adds quite a bit of complexity
        # https://github.com/DLR-RM/stable-baselines3/pull/243#discussion_r531535702
        self.optimize_memory_usage = optimize_memory_usage

        self.observations = {
            key: jnp.zeros((self.buffer_size, self.n_envs, *_obs_shape), dtype=observation_space[key].dtype)
            for key, _obs_shape in self.obs_shape.items()
        }
        self.next_observations = {
            key: jnp.zeros((self.buffer_size, self.n_envs, *_obs_shape), dtype=observation_space[key].dtype)
            for key, _obs_shape in self.obs_shape.items()
        }

        self.actions = jnp.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=action_space.dtype)
        self.rewards = jnp.zeros((self.buffer_size, self.n_envs), dtype=jnp.float32)
        self.dones = jnp.zeros((self.buffer_size, self.n_envs), dtype=jnp.float32)

        # Handle timeouts termination properly if needed
        # see https://github.com/DLR-RM/stable-baselines3/issues/284
        self.handle_timeout_termination = handle_timeout_termination
        self.timeouts = jnp.zeros((self.buffer_size, self.n_envs), dtype=jnp.float32)

        if psutil is not None:
            obs_nbytes = 0
            for _, obs in self.observations.items():
                obs_nbytes += obs.nbytes

            total_memory_usage = obs_nbytes + self.actions.nbytes + self.rewards.nbytes + self.dones.nbytes
            if self.next_observations is not None:
                next_obs_nbytes = 0
                for _, obs in self.observations.items():
                    next_obs_nbytes += obs.nbytes
                total_memory_usage += next_obs_nbytes

            if total_memory_usage > mem_available:
                # Convert to GB
                total_memory_usage /= 1e9
                mem_available /= 1e9
                warnings.warn(
                    "This system does not have apparently enough memory to store the complete "
                    f"replay buffer {total_memory_usage:.2f}GB > {mem_available:.2f}GB"
                )

    def add(
        self,
        obs: Dict[str, jnp.ndarray],
        next_obs: Dict[str, jnp.ndarray],
        action: jnp.ndarray,
        reward: jnp.ndarray,
        done: jnp.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:  # pytype: disable=signature-mismatch
        # Copy to avoid modification by reference
        for key in self.observations.keys():
            # Reshape needed when using multiple envs with discrete observations
            # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
            if isinstance(self.observation_space.spaces[key], spaces.Discrete):
                obs[key] = obs[key].reshape((self.n_envs,) + self.obs_shape[key])
            self.observations[key][self.pos] = jnp.array(obs[key])

        for key in self.next_observations.keys():
            if isinstance(self.observation_space.spaces[key], spaces.Discrete):
                next_obs[key] = next_obs[key].reshape((self.n_envs,) + self.obs_shape[key])
            self.next_observations[key][self.pos] = jnp.array(next_obs[key]).copy()

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        self.actions[self.pos] = jnp.array(action).copy()
        self.rewards[self.pos] = jnp.array(reward).copy()
        self.dones[self.pos] = jnp.array(done).copy()

        if self.handle_timeout_termination:
            self.timeouts[self.pos] = jnp.array([info.get("TimeLimit.truncated", False) for info in infos])

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(
        self,
        batch_size: int,
        env: Optional[VecNormalize] = None,
    ) -> DictReplayBufferSamples:  # type: ignore[signature-mismatch] #FIXME:
        """
        Sample elements from the replay buffer.

        :param batch_size: Number of element to sample
        :param env: associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
        """
        return super(ReplayBuffer, self).sample(batch_size=batch_size, env=env)

    def _get_samples(
        self,
        batch_inds: jnp.ndarray,
        env: Optional[VecNormalize] = None,
    ) -> DictReplayBufferSamples:  # type: ignore[signature-mismatch] #FIXME:
        # Sample randomly the env idx
        env_indices = jnp.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        # Normalize if needed and remove extra dimension (we are using only one env for now)
        obs_ = self._normalize_obs({key: obs[batch_inds, env_indices, :] for key, obs in self.observations.items()}, env)
        next_obs_ = self._normalize_obs(
            {key: obs[batch_inds, env_indices, :] for key, obs in self.next_observations.items()}, env
        )

        # Convert to torch tensor
        observations = {key: self.to_torch(obs) for key, obs in obs_.items()}
        next_observations = {key: self.to_torch(obs) for key, obs in next_obs_.items()}

        return DictReplayBufferSamples(
            observations=observations,
            actions=self.to_torch(self.actions[batch_inds, env_indices]),
            next_observations=next_observations,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            dones=self.to_torch(self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(
                -1, 1
            ),
            rewards=self.to_torch(self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env)),
        )


class DictRolloutBuffer(RolloutBuffer):
    """
    Dict Rollout buffer used in on-policy algorithms like A2C/PPO.
    Extends the RolloutBuffer to use dictionary observations

    It corresponds to ``buffer_size`` transitions collected
    using the current policy.
    This experience will be discarded after the policy update.
    In order to use PPO objective, we also store the current value of each state
    and the log probability of each taken action.

    The term rollout here refers to the model-free notion and should not
    be used with the concept of rollout used in model-based RL or planning.
    Hence, it is only involved in policy and value function training but not action selection.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to Monte-Carlo advantage estimate when set to 1.
    :param gamma: Discount factor
    :param n_envs: Number of parallel environments
    """

    observations: Dict[str, jnp.ndarray]

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
    ):
        super(RolloutBuffer, self).__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)

        assert isinstance(self.obs_shape, dict), "DictRolloutBuffer must be used with Dict obs space only"

        self.gae_lambda = gae_lambda
        self.gamma = gamma

        self.generator_ready = False
        self.reset()

    def reset(self) -> None:
        assert isinstance(self.obs_shape, dict), "DictRolloutBuffer must be used with Dict obs space only"
        self.observations = {}
        for key, obs_input_shape in self.obs_shape.items():
            self.observations[key] = [jnp.zeros((self.n_envs, *obs_input_shape), dtype=jnp.float32)] * self.buffer_size
        self.actions = [jnp.zeros(( self.n_envs, self.action_dim), dtype=jnp.float32)] * self.buffer_size
        self.rewards =  [jnp.zeros(( self.n_envs), dtype=jnp.float32)] * self.buffer_size
        self.returns =  [jnp.zeros(( self.n_envs), dtype=jnp.float32)] * self.buffer_size
        self.episode_starts =  [jnp.zeros(( self.n_envs), dtype=jnp.float32)] * self.buffer_size
        self.values =  [jnp.zeros(( self.n_envs), dtype=jnp.float32)] * self.buffer_size
        self.log_probs =  [jnp.zeros(( self.n_envs), dtype=jnp.float32)] * self.buffer_size
        self.advantages =  [jnp.zeros(( self.n_envs), dtype=jnp.float32)] * self.buffer_size
        self.generator_ready = False
        super(RolloutBuffer, self).reset()

    def add(
        self,
        obs: Dict[str, jnp.ndarray],
        action: jnp.ndarray,
        reward: jnp.ndarray,
        episode_start: jnp.ndarray,
        value: jnp.ndarray,
        log_prob:  jnp.ndarray,
    ) -> None:  # pytype: disable=signature-mismatch
        """
        :param obs: Observation
        :param action: Action
        :param reward:
        :param episode_start: Start of episode signal.
        :param value: estimated value of the current state
            following the current policy.
        :param log_prob: log probability of the action
            following the current policy.
        """
        if len(log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = log_prob.reshape(-1, 1)

        for key in self.observations.keys():
            obs_ = jnp.array(obs[key]).copy()
            # Reshape needed when using multiple envs with discrete observations
            # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
            if isinstance(self.observation_space.spaces[key], spaces.Discrete):
                obs_ = obs_.reshape((self.n_envs,) + self.obs_shape[key])
            self.observations[key][self.pos] = obs_

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        self.actions[self.pos] = jnp.array(action).copy()
        self.rewards[self.pos] = jnp.array(reward).copy()
        self.episode_starts[self.pos] = jnp.array(episode_start).copy()
        self.values[self.pos] = value.clone().cpu().numpy().flatten()
        self.log_probs[self.pos] = log_prob.clone().cpu().numpy()
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def get(
        self,
            random_key,
        batch_size: Optional[int] = None,
    ) -> Generator[DictRolloutBufferSamples, None, None]:  # type: ignore[signature-mismatch] #FIXME
        assert self.full, ""
        indices = jax.random.permutation(random_key,self.buffer_size * self.n_envs)
        # Prepare the data
        if not self.generator_ready:
            for key, obs in self.observations.items():
                self.observations[key] = self.swap_and_flatten(obs)

            _tensor_names = ["actions", "values", "log_probs", "advantages", "returns"]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(
        self,
        batch_inds: jnp.ndarray,
        env: Optional[VecNormalize] = None,
    ) -> DictRolloutBufferSamples:  # type: ignore[signature-mismatch] #FIXME
        return DictRolloutBufferSamples(
            observations={key: self.to_torch(obs[batch_inds]) for (key, obs) in self.observations.items()},
            actions=self.to_torch(self.actions[batch_inds]),
            old_values=self.to_torch(self.values[batch_inds].flatten()),
            old_log_prob=self.to_torch(self.log_probs[batch_inds].flatten()),
            advantages=self.to_torch(self.advantages[batch_inds].flatten()),
            returns=self.to_torch(self.returns[batch_inds].flatten()),
        )