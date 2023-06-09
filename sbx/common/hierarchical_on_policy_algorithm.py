from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union
import sys
import time
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import spaces
from sbx.common.buffers import HierarchicalRolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback

from sbx.common.utils import safe_mean
from sbx.common.policies import BasePolicy
from sbx.common.base_class import BaseAlgorithmJax
from sbx.common.policies import HierarchicalBaseJaxPolicy

from sbx.common.type_aliases import GymEnv, Schedule, MaybeCallback
from sbx.common.gymanax_wrapper import GymnaxToVectorGymWrapper as VecEnv
OnPolicyAlgorithmJaxSelf = TypeVar("HierarchicalOnPolicyAlgorithmJaxSelf", bound="HierarchicalOnPolicyAlgorithmJax")


from sbx.common import utils
from collections import deque
class HierarchicalOnPolicyAlgorithmJax(BaseAlgorithmJax):
    rollout_buffer: HierarchicalRolloutBuffer
    policy: HierarchicalBaseJaxPolicy  # type: ignore[assignment]

    def _setup_learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        reset_num_timesteps: bool = True,
        tb_log_name: str = "run",
        progress_bar: bool = False,
    ) -> Tuple[int, BaseCallback]:
        """
        Initialize different variables needed for training.

        :param total_timesteps: The total number of samples (env steps) to train on
        :param callback: Callback(s) called at every step with state of the algorithm.
        :param reset_num_timesteps: Whether to reset or not the ``num_timesteps`` attribute
        :param tb_log_name: the name of the run for tensorboard log
        :param progress_bar: Display a progress bar using tqdm and rich.
        :return: Total timesteps and callback(s)
        """
        self.start_time = time.time_ns()

        if self.ep_info_buffer is None or reset_num_timesteps:
            # Initialize buffers if they don't exist, or reinitialize if resetting counters
            self.ep_info_buffer = deque(maxlen=self._stats_window_size)
            self.ep_success_buffer = deque(maxlen=self._stats_window_size)

        if self.action_noise is not None:
            self.action_noise.reset()

        if reset_num_timesteps:
            self.num_timesteps = 0
            self._episode_num = 0
        else:
            # Make sure training timesteps are ahead of the internal counter
            total_timesteps += self.num_timesteps
        self._total_timesteps = total_timesteps
        self._num_timesteps_at_start = self.num_timesteps

        # Avoid resetting the environment when calling ``.learn()`` consecutive times
        if reset_num_timesteps or self._last_obs is None:
            assert self.env is not None
            # pytype: disable=annotation-type-mismatch
            self._last_obs = self.env.reset()  # type: ignore[assignment]
            # pytype: enable=annotation-type-mismatch
            self._last_episode_starts = jnp.ones((self.env.num_envs,), dtype=bool)
            self._last_option_starts = jnp.ones((self.env.num_envs,), dtype=bool)
            self._last_options = jnp.zeros((self.env.num_envs,), dtype=int)
            # Retrieve unnormalized observation for saving into the buffer
            if self._vec_normalize_env is not None:
                self._last_original_obs = self._vec_normalize_env.get_original_obs()

        # Configure logger's outputs if no logger was passed
        if not self._custom_logger:
            self._logger = utils.configure_logger(self.verbose, self.tensorboard_log, tb_log_name, reset_num_timesteps)

        # Create eval callback if needed
        callback = self._init_callback(callback, progress_bar)

        return total_timesteps, callback
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
        device: str = "auto",
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
        )
        self.n_steps = n_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.n_options = policy_kwargs["net_arch"]["n_options"]
        # Will be updated later
        self.key = jax.random.PRNGKey(0)

        if _init_setup_model:
            self._setup_model()
    def _get_torch_save_params(self):
        return [], []

    def _excluded_save_params(self) -> List[str]:
        excluded = super()._excluded_save_params()
        excluded.remove("policy")
        return excluded

    def set_random_seed(self, seed: Optional[int]) -> None:  # type: ignore[override]
        super().set_random_seed(seed)
        if seed is None:
            # Sample random seed
            seed = np.random.randint(2**14)
        self.key = jax.random.PRNGKey(seed)

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        self.rollout_buffer = HierarchicalRolloutBuffer(
            self.n_steps,
            self.observation_space,
            self.action_space,
            self.n_options,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
            device="cpu",  # force cpu device to easy torch -> numpy conversion
        )

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: HierarchicalRolloutBuffer,
        n_rollout_steps: int,
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
        while n_steps < n_rollout_steps:
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
            actions, options,  log_probs, option_log_probs, values, option_values = self.policy.predict_all(obs_jnp,self._last_options,self._last_option_starts,
                                                                 self.policy.noise_key)

         #   actions = jnp.array(actions)
         #   log_probs = jnp.array(log_probs)
        #    values = jnp.array(values)

            # Rescale and perform action
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, gym.spaces.Box):
                clipped_actions = jnp.clip(actions, self.action_space.low, self.action_space.high)
            else:
                clipped_actions = actions
       #     print ("clipped_actions",clipped_actions)
         #   print ("env_state",env.env_state)
            new_obs, rewards, dones, infos = env.step(clipped_actions)
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

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
       #     if infos.get("terminal_observation") is not None and infos.get("TimeLimit.truncated", False):
       #         terminal_obs = self.policy.prepare_obs(infos["terminal_observation"])[0]
        #        terminal_value = jnp.array(
         #           self.vf.apply(  # type: ignore[union-attr]
          #              self.policy.vf_state.params,
           #             terminal_obs,
            #        ).flatten()
             #   )
              #  rewards += self.gamma * terminal_value

            rollout_buffer.add(
                self._last_obs,  # type: ignore
                actions,
                rewards,
                self._last_episode_starts,  # type: ignore
                values,
                options,
                self._last_option_starts,
                option_values,
                log_probs,
                option_log_probs,
                option_start_log_probs
            )

            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones
            self._last_options = options
            self._last_option_starts, option_start_log_probs, option_start_logits = self.policy.option_start(new_obs, options,dones,self.policy.noise_key)

        last_values = self.policy.value_function(new_obs)
        last_option_values = self.policy.option_value_function(new_obs, options)
        rollout_buffer.compute_returns_and_advantage(last_values=last_values, last_option_values=last_option_values,
                                                     last_option_start_log_prob=option_start_logits, dones=dones)

        callback.on_rollout_end()

        return True

    def train(self) -> None:
        """
        Consume current rollout data and update policy parameters.
        Implemented by individual algorithms.
        """
        raise NotImplementedError

    def learn(
        self: OnPolicyAlgorithmJaxSelf,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "HierarchicalOnPolicyAlgorithmJax",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> OnPolicyAlgorithmJaxSelf:
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


    def unsupervised_learn(
        self: OnPolicyAlgorithmJaxSelf,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "UnsupervisedHierachicalOnPolicyAlgorithmJax",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> OnPolicyAlgorithmJaxSelf:
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

            self.unsupervised_train()

        callback.on_training_end()

        return self