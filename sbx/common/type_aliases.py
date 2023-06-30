import flax
from flax.training.train_state import TrainState

import jax.numpy as jnp
import sys
from enum import Enum
from typing import Any, Callable, Dict, List, NamedTuple, Optional, SupportsFloat, Tuple, Union

if sys.version_info >= (3, 8):
    from typing import Protocol
else:
    from typing_extensions import Protocol

from stable_baselines3.common import callbacks, vec_env
class RLTrainState(TrainState):  # type: ignore[misc]
    target_params: flax.core.FrozenDict  # type: ignore[misc]

from sbx.common.gymanax_wrapper import GymnaxToVectorGymWrapper as VecEnv
from sbx.common.gymanax_wrapper import GymnaxToGymWrapper as Env
"""Common aliases for type hints"""

GymEnv = Union[Env, VecEnv]
GymObs = Union[Tuple, Dict[str, Any], jnp.ndarray, int]
GymResetReturn = Tuple[GymObs, Dict]
AtariResetReturn = Tuple[jnp.ndarray, Dict[str, Any]]
GymStepReturn = Tuple[GymObs, float, bool, bool, Dict]
AtariStepReturn = Tuple[jnp.ndarray, SupportsFloat, bool, bool, Dict[str, Any]]
JnpDict = Dict[str, jnp.ndarray]
OptimizerStateDict = Dict[str, Any]
MaybeCallback = Union[None, Callable, List[callbacks.BaseCallback], callbacks.BaseCallback]

# A schedule takes the remaining progress as input
# and ouputs a scalar (e.g. learning rate, clip range, ...)
Schedule = Callable[[float], float]


class ReplayBufferSamplesNp(NamedTuple):
    observations: jnp.ndarray
    actions: jnp.ndarray
    next_observations: jnp.ndarray
    dones: jnp.ndarray
    rewards: jnp.ndarray

class RolloutBufferSamples(NamedTuple):
    observations: jnp.ndarray
    actions: jnp.ndarray
    old_values: jnp.ndarray
    old_log_prob: jnp.ndarray
    advantages: jnp.ndarray
    returns: jnp.ndarray

class HierarchicalRolloutBufferSamples(NamedTuple):
    observations: jnp.ndarray
    actions: jnp.ndarray
    options: jnp.ndarray
    option_starts: jnp.ndarray
    episode_starts: jnp.ndarray
    last_options: jnp.ndarray
    old_log_probs: jnp.ndarray
    old_option_log_probs: jnp.ndarray
    old_option_start_log_probs: jnp.ndarray
    advantages: jnp.ndarray
    option_advantages: jnp.ndarray
    option_start_advantages: jnp.ndarray
    returns: jnp.ndarray

class UnsupervisedHierarchicalRolloutBufferSamples(NamedTuple):
    observations: jnp.ndarray
    actions: jnp.ndarray
    options: jnp.ndarray
    option_starts: jnp.ndarray
    episode_starts: jnp.ndarray
    last_options: jnp.ndarray
    old_log_probs: jnp.ndarray
    old_option_log_probs: jnp.ndarray
    old_option_start_log_probs: jnp.ndarray
    advantages: jnp.ndarray
    option_advantages: jnp.ndarray
    option_start_advantages: jnp.ndarray
    returns: jnp.ndarray
    variational_log_posterior: jnp.ndarray
    control_returns: jnp.ndarray
    option_start_time: jnp.ndarray
    entropy: jnp.ndarray
    option_rewards: jnp.ndarray
    option_returns: jnp.ndarray

class DictRolloutBufferSamples(NamedTuple):
    observations: JnpDict
    actions: jnp.ndarray
    old_values: jnp.ndarray
    old_log_prob: jnp.ndarray
    advantages: jnp.ndarray
    returns: jnp.ndarray


class ReplayBufferSamples(NamedTuple):
    observations: jnp.ndarray
    actions: jnp.ndarray
    next_observations: jnp.ndarray
    dones: jnp.ndarray
    rewards: jnp.ndarray


class DictReplayBufferSamples(NamedTuple):
    observations: JnpDict
    actions: jnp.ndarray
    next_observations: JnpDict
    dones: jnp.ndarray
    rewards: jnp.ndarray


class RolloutReturn(NamedTuple):
    episode_timesteps: int
    n_episodes: int
    continue_training: bool


class TrainFrequencyUnit(Enum):
    STEP = "step"
    EPISODE = "episode"


class TrainFreq(NamedTuple):
    frequency: int
    unit: TrainFrequencyUnit  # either "step" or "episode"


class PolicyPredictor(Protocol):
    def predict(
        self,
        observation: Union[jnp.ndarray, Dict[str, jnp.ndarray]],
        state: Optional[Tuple[jnp.ndarray, ...]] = None,
        episode_start: Optional[jnp.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[jnp.ndarray, Optional[Tuple[jnp.ndarray, ...]]]:
        """
        Get the policy action from an observation (and optional hidden state).
        Includes sugar-coating to handle different observations (e.g. normalizing images).

        :param observation: the input observation
        :param state: The last hidden states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
            this correspond to beginning of episodes,
            where the hidden states of the RNN must be reset.
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next hidden state
            (used in recurrent policies)
        """