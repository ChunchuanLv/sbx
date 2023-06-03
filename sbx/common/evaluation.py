import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import jax.numpy as jnp
import jax
from sbx.common import type_aliases
from sbx.common.type_aliases import Env,VecEnv
from sbx.common.gymanax_wrapper import GymnaxToVectorGymWrapper
from tqdm import tqdm
def evaluate_policy(
    model: type_aliases.PolicyPredictor,
    env: Union[Env, VecEnv],
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
    callback: Optional[Callable[[Dict[str, Any], Dict[str, Any]], None]] = None,
    reward_threshold: Optional[float] = None,
    return_episode_rewards: bool = False,
    warn: bool = True,
) -> Union[Tuple[float, float], Tuple[List[float], List[int]]]:
    """
    Runs policy for ``n_eval_episodes`` episodes and returns average reward.
    If a vector env is passed in, this divides the episodes to evaluate onto the
    different elements of the vector env. This static division of work is done to
    remove bias. See https://github.com/DLR-RM/stable-baselines3/issues/402 for more
    details and discussion.

    .. note::
        If environment has not been wrapped with ``Monitor`` wrapper, reward and
        episode lengths are counted as it appears with ``env.step`` calls. If
        the environment contains wrappers that modify rewards or episode lengths
        (e.g. reward scaling, early episode reset), these will affect the evaluation
        results as well. You can avoid this by wrapping environment with ``Monitor``
        wrapper before anything else.

    :param model: The RL agent you want to evaluate. This can be any object
        that implements a `predict` method, such as an RL algorithm (``BaseAlgorithm``)
        or policy (``BasePolicy``).
    :param env: The gym environment or ``VecEnv`` environment.
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param deterministic: Whether to use deterministic or stochastic actions
    :param render: Whether to render the environment or not
    :param callback: callback function to do additional checks,
        called after each step. Gets locals() and globals() passed as parameters.
    :param reward_threshold: Minimum expected reward per episode,
        this will raise an error if the performance is not met
    :param return_episode_rewards: If True, a list of rewards and episode lengths
        per episode will be returned instead of the mean.
    :param warn: If True (default), warns user about lack of a Monitor wrapper in the
        evaluation environment.
    :return: Mean reward per episode, std of reward per episode.
        Returns ([float], [int]) when ``return_episode_rewards`` is True, first
        list containing per-episode rewards and second containing per-episode lengths
        (in number of steps).
    """
    is_monitor_wrapped = False
    # Avoid circular import
    from stable_baselines3.common.monitor import Monitor

    if not isinstance(env, VecEnv):
        env = GymnaxToVectorGymWrapper(env)

 #   is_monitor_wrapped = is_vecenv_wrapped(env, VecMonitor) or env.env_is_wrapped(Monitor)[0]

    if not is_monitor_wrapped and warn:
        warnings.warn(
            "Evaluation environment is not wrapped with a ``Monitor`` wrapper. "
            "This may result in reporting modified episode lengths and rewards, if other wrappers happen to modify these. "
            "Consider wrapping environment first with ``Monitor`` wrapper.",
            UserWarning,
        )

    n_envs = env.num_envs
    episode_rewards = []
    episode_lengths = []

    episode_counts = jnp.zeros(n_envs, dtype=jnp.int32)
    # Divides episodes among different sub environments in the vector as evenly as possible
    episode_count_targets = jnp.array([(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype=jnp.int32)

    current_rewards = jnp.zeros(n_envs)
    current_lengths = jnp.zeros(n_envs, dtype=jnp.int32)
    observations = env.reset()
    states = None
    episode_starts = jnp.ones((env.num_envs,), dtype=bool)

    progress = tqdm(total=n_eval_episodes, desc="Evaluating")
    while (episode_counts < episode_count_targets).any():
        actions, states = model.predict(
            observations,  # type: ignore[arg-type]
            state=states,
            episode_start=episode_starts,
            deterministic=deterministic,
        )
        new_observations, rewards, dones, infos = env.step(actions)
        current_rewards += rewards
        current_lengths += 1
        env_not_finished = episode_counts < episode_count_targets

        # unpack values so that the callback can access the local variables
        episode_starts = dones

        new_counts = jnp.logical_and(env_not_finished, dones)
        progress.update(jnp.sum(new_counts).item())
        episode_counts = jax.lax.select(new_counts, episode_counts + 1,episode_counts)
        if callback is not None:
            callback(locals(), globals())

        episode_rewards.append(current_rewards[new_counts])
        episode_lengths.append(current_lengths[new_counts])
        current_rewards = jax.lax.select(new_counts, jnp.zeros(n_envs), current_rewards)
        current_lengths = jax.lax.select(new_counts, jnp.zeros(n_envs, dtype=jnp.int32), current_lengths)

        observations = new_observations

        progress.set_description("Mean Reward %f" % jnp.concatenate(episode_lengths).mean().item())
    #    print ((episode_counts < episode_count_targets).any())
    #    print (len(episode_lengths))
     #   print (len(episode_rewards))

        if render:
            env.render()
    episode_lengths = jnp.concatenate(episode_lengths)
    episode_rewards = jnp.concatenate(episode_rewards)
    mean_reward = jnp.mean(episode_rewards)
    std_reward = jnp.std(episode_rewards)
    if reward_threshold is not None:
        assert mean_reward > reward_threshold, "Mean reward below threshold: " f"{mean_reward:.2f} < {reward_threshold:.2f}"
    if return_episode_rewards:
        return episode_rewards, episode_lengths
    return mean_reward, std_reward


def evaluate_hierarchical_policy(
    model: type_aliases.PolicyPredictor,
    env: Union[Env, VecEnv],
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
    callback: Optional[Callable[[Dict[str, Any], Dict[str, Any]], None]] = None,
    reward_threshold: Optional[float] = None,
    return_episode_rewards: bool = False,
    warn: bool = True,
) -> Union[Tuple[float, float], Tuple[List[float], List[int]]]:
    """
    Runs policy for ``n_eval_episodes`` episodes and returns average reward.
    If a vector env is passed in, this divides the episodes to evaluate onto the
    different elements of the vector env. This static division of work is done to
    remove bias. See https://github.com/DLR-RM/stable-baselines3/issues/402 for more
    details and discussion.

    .. note::
        If environment has not been wrapped with ``Monitor`` wrapper, reward and
        episode lengths are counted as it appears with ``env.step`` calls. If
        the environment contains wrappers that modify rewards or episode lengths
        (e.g. reward scaling, early episode reset), these will affect the evaluation
        results as well. You can avoid this by wrapping environment with ``Monitor``
        wrapper before anything else.

    :param model: The RL agent you want to evaluate. This can be any object
        that implements a `predict` method, such as an RL algorithm (``BaseAlgorithm``)
        or policy (``BasePolicy``).
    :param env: The gym environment or ``VecEnv`` environment.
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param deterministic: Whether to use deterministic or stochastic actions
    :param render: Whether to render the environment or not
    :param callback: callback function to do additional checks,
        called after each step. Gets locals() and globals() passed as parameters.
    :param reward_threshold: Minimum expected reward per episode,
        this will raise an error if the performance is not met
    :param return_episode_rewards: If True, a list of rewards and episode lengths
        per episode will be returned instead of the mean.
    :param warn: If True (default), warns user about lack of a Monitor wrapper in the
        evaluation environment.
    :return: Mean reward per episode, std of reward per episode.
        Returns ([float], [int]) when ``return_episode_rewards`` is True, first
        list containing per-episode rewards and second containing per-episode lengths
        (in number of steps).
    """
    is_monitor_wrapped = False
    # Avoid circular import
    from stable_baselines3.common.monitor import Monitor

    if not isinstance(env, VecEnv):
        env = GymnaxToVectorGymWrapper(env)

 #   is_monitor_wrapped = is_vecenv_wrapped(env, VecMonitor) or env.env_is_wrapped(Monitor)[0]

    if not is_monitor_wrapped and warn:
        warnings.warn(
            "Evaluation environment is not wrapped with a ``Monitor`` wrapper. "
            "This may result in reporting modified episode lengths and rewards, if other wrappers happen to modify these. "
            "Consider wrapping environment first with ``Monitor`` wrapper.",
            UserWarning,
        )

    n_envs = env.num_envs
    episode_rewards = []
    episode_lengths = []

    episode_counts = jnp.zeros(n_envs, dtype=jnp.int32)
    # Divides episodes among different sub environments in the vector as evenly as possible
    episode_count_targets = jnp.array([(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype=jnp.int32)

    current_rewards = jnp.zeros(n_envs)
    current_lengths = jnp.zeros(n_envs, dtype=jnp.int32)
    observations = env.reset()
    states = None
    episode_starts = jnp.ones((env.num_envs,), dtype=bool)
    option_starts = jnp.ones((env.num_envs,), dtype=bool)
    options = jnp.zeros((env.num_envs,1), dtype=int)

    progress = tqdm(total=n_eval_episodes, desc="Evaluating")
    while (episode_counts < episode_count_targets).any():
        actions, states = model.predict(
            observations,  # type: ignore[arg-type]
            options,  # type: ignore[arg-type]
            state=states,
            episode_start=episode_starts,
            option_starts=option_starts,
            deterministic=deterministic,
        )
        new_observations, rewards, dones, infos = env.step(actions)
        current_rewards += rewards
        current_lengths += 1
        env_not_finished = episode_counts < episode_count_targets

        # unpack values so that the callback can access the local variables
        episode_starts = dones

        new_counts = jnp.logical_and(env_not_finished, dones)
        progress.update(jnp.sum(new_counts).item())
        episode_counts = jax.lax.select(new_counts, episode_counts + 1,episode_counts)
        if callback is not None:
            callback(locals(), globals())

        episode_rewards.append(current_rewards[new_counts])
        episode_lengths.append(current_lengths[new_counts])
        current_rewards = jax.lax.select(new_counts, jnp.zeros(n_envs), current_rewards)
        current_lengths = jax.lax.select(new_counts, jnp.zeros(n_envs, dtype=jnp.int32), current_lengths)

        observations = new_observations

        progress.set_description("Mean Reward %f" % jnp.concatenate(episode_lengths).mean().item())
    #    print ((episode_counts < episode_count_targets).any())
    #    print (len(episode_lengths))
     #   print (len(episode_rewards))

        if render:
            env.render()
    episode_lengths = jnp.concatenate(episode_lengths)
    episode_rewards = jnp.concatenate(episode_rewards)
    mean_reward = jnp.mean(episode_rewards)
    std_reward = jnp.std(episode_rewards)
    if reward_threshold is not None:
        assert mean_reward > reward_threshold, "Mean reward below threshold: " f"{mean_reward:.2f} < {reward_threshold:.2f}"
    if return_episode_rewards:
        return episode_rewards, episode_lengths
    return mean_reward, std_reward