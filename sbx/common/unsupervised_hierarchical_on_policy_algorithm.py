from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union
import sys
import time
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import spaces
from sbx.common.buffers import UnsupervisedHierarchicalRolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback

from sbx.common.utils import safe_mean
from sbx.common.policies import BasePolicy
from sbx.common.hierarchical_on_policy_algorithm import HierarchicalOnPolicyAlgorithmJax
from sbx.common.policies import UnsupervisedHierarchicalBaseJaxPolicy

from sbx.common.type_aliases import GymEnv, Schedule, MaybeCallback
from sbx.common.gymanax_wrapper import GymnaxToVectorGymWrapper as VecEnv
UnsupervisedHierarchicalOnPolicyAlgorithmJaxSelf = TypeVar("UnsupervisedHierarchicalOnPolicyAlgorithmJaxSelf", bound="UnsupervisedHierarchicalOnPolicyAlgorithmJax")


class UnsupervisedHierarchicalOnPolicyAlgorithmJax(HierarchicalOnPolicyAlgorithmJax):
    rollout_buffer: UnsupervisedHierarchicalRolloutBuffer
    policy: UnsupervisedHierarchicalBaseJaxPolicy  # type: ignore[assignment]
    intri_coef:float
    def __init__(
        self,
        policy: Union[str, Type[BasePolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule],
        n_steps: int,
        gamma: float,
        gae_lambda: float,
        ent_coef: float,
        vf_coef: float,
        max_grad_norm: float,
        use_sde: bool,
        sde_sample_freq: int,
        support_multi_env: bool = False,
        tensorboard_log: Optional[str] = None,
        monitor_wrapper: bool = True,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        env_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        _init_setup_model: bool = True,
        supported_action_spaces: Optional[Tuple[Type[spaces.Space], ...]] = None,
    ):
        super().__init__(
            policy=policy,  # type: ignore[arg-type]
            env=env,
            learning_rate=learning_rate,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            monitor_wrapper=monitor_wrapper,
            support_multi_env = support_multi_env,
            policy_kwargs=policy_kwargs,
            env_kwargs=env_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            seed=seed,
            supported_action_spaces=supported_action_spaces,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
        )
    def set_random_seed(self, seed: Optional[int]) -> None:  # type: ignore[override]
        super().set_random_seed(seed)
        if seed is None:
            # Sample random seed
            seed = np.random.randint(2**14)
        self.key = jax.random.PRNGKey(seed)

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        self.rollout_buffer = UnsupervisedHierarchicalRolloutBuffer(
            self.n_steps,
            self.observation_space,
            self.action_space,
            self.n_options,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: UnsupervisedHierarchicalRolloutBuffer,
        n_rollout_steps: int,
        unsupervised:bool = True
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_rollout_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"  # type: ignore[has-type]
        # Switch to eval mode (this affects batch norm / dropout)

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise()

        option_start_log_probs = jnp.zeros((self.env.num_envs,), dtype=float)
        option_start_time = jnp.zeros((self.env.num_envs,), dtype=int)
        self._last_episode_starts = jnp.ones((self.env.num_envs,), dtype=bool)
        while n_steps < n_rollout_steps:
            option_start_time = jnp.where(self._last_option_starts,n_steps * jnp.ones((self.env.num_envs,), dtype=int),option_start_time)
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise()

            if not self.use_sde or isinstance(self.action_space, gym.spaces.Discrete):
                # Always sample new stochastic action
                self.policy.reset_noise()
            obs_jnp, vectorized_env = self.policy.prepare_obs(self._last_obs)  # type: ignore[has-type]
        #    print ("obs_jnp",obs_jnp)
        #    print ("last_options",self._last_options)
        #    print ("last_option_starts",self._last_option_starts)
        #    print ("noise_key", self.policy.noise_key)
            actions, self._last_option_starts, options,  log_probs, option_log_probs, values, option_values,last_option_values,\
                option_reward_value,  last_option_control_values,control_values,entropy, variational_log_posteriors, option_start_log_probs\
                = self.policy.predict_all(obs_jnp,self._last_options,self._last_episode_starts,
                                                                 self.policy.noise_key)

            # Rescale and perform action
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, gym.spaces.Box):
                clipped_actions = jnp.clip(actions, self.action_space.low, self.action_space.high)
            else:
                clipped_actions = actions
            new_obs, extrin_rewards, dones, infos = env.step(clipped_actions)
            extrin_rewards = extrin_rewards * (1-self.intri_coef)#jnp.zeros((self.env.num_envs,), dtype=float)
            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False
            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, gym.spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

         #   variational_posterior_probs = jnp.exp(variational_log_posteriors) * (1 - self._last_episode_starts)
        #    weighted_variational_log_posterior = variational_log_posteriors * self.gamma ** (option_start_time - n_steps)
       #     weighted_variational_log_posterior = jnp.clip(weighted_variational_log_posterior,-1e2,1e2)
      #      intrin_reward = self._last_option_starts * ( entropy + weighted_variational_log_posterior * (1 - self._last_episode_starts))

            rollout_buffer.add(
                self._last_obs,  # type: ignore
                actions,
                extrin_rewards,
                self._last_episode_starts,  # type: ignore
                values,
                options,
                self._last_option_starts,
                option_values,
                last_option_values,
                log_probs,
                option_log_probs,
                option_start_log_probs,
                variational_log_posteriors,
                entropy,
                control_values,
                last_option_control_values,
                option_start_time,
                option_reward_value,
            )

            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones
        self._last_options = options
        self._last_option_starts, option_start_log_probs = self.policy.option_start(new_obs, options,dones,self.policy.noise_key)

        last_values = self.policy.value_function(new_obs)
        last_option_values = self.policy.option_value_function(new_obs, options)
        last_control_values = self.policy.control_value_function(new_obs, options)
        last_entropy = self.policy.policy_entropy(new_obs)
        last_variational_log_posterior = self.policy.variational_posterior_log_prob(new_obs, options)

        option_start_probs = jnp.exp(option_start_log_probs)
        rollout_buffer.compute_returns_and_advantage(last_values=last_values,last_option_values=last_option_values,
                                                     last_control_value=last_control_values,
                                                     last_variational_log_posterior=last_variational_log_posterior,
                                                     last_option_start_probs=option_start_probs,dones=dones,
                                                     intri_coeffecient=self.intri_coef)

        callback.on_rollout_end()

        return True

    def train(self) -> None:
        """
        Consume current rollout data and update policy parameters.
        Implemented by individual algorithms.
        """
        raise NotImplementedError

    def learn(
        self: UnsupervisedHierarchicalOnPolicyAlgorithmJaxSelf ,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "HierarchicalOnPolicyAlgorithmJax",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> UnsupervisedHierarchicalOnPolicyAlgorithmJaxSelf:
        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        callback.on_training_start(locals(), globals())

        assert self.env is not None

        while self.num_timesteps < total_timesteps:
            continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer, n_rollout_steps=self.n_steps)

            if continue_training is False:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                assert self.ep_info_buffer is not None
                time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
                fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
                self.logger.record("time/iterations", iteration, exclude="tensorboard")
                if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                    self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
                    self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
                self.logger.record("time/fps", fps)
                self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
                self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
                self.logger.dump(step=self.num_timesteps)

            self.train()

        callback.on_training_end()

        return self

