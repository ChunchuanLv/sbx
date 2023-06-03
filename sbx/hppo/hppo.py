import warnings
from functools import partial
from typing import Any, Dict, Optional, Type, TypeVar, Union,Tuple

import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
from gymnax.environments import spaces
from sbx.common.type_aliases import GymEnv, MaybeCallback, Schedule
from sbx.common.utils import explained_variance, get_schedule_fn

from sbx.common.hierarchical_on_policy_algorithm import HierarchicalOnPolicyAlgorithmJax
from sbx.hppo.policies import OptionActor, OptionStarter, Actor, Critic, HPPOPolicy
HPPOSelf = TypeVar("HPPOSelf", bound="HPPO")


class HPPO(HierarchicalOnPolicyAlgorithmJax):
    """
    Proximal Policy Optimization algorithm (PPO) (clip version)

    Paper: https://arxiv.org/abs/1707.06347
    Code: This implementation borrows code from OpenAI Spinning Up (https://github.com/openai/spinningup/)
    https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail and
    Stable Baselines (PPO2 from https://github.com/hill-a/stable-baselines)

    Introduction to PPO: https://spinningup.openai.com/en/latest/algorithms/ppo.html

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. rollout buffer size is n_steps * n_envs where n_envs is number of environment copies running in parallel)
        NOTE: n_steps * n_envs must be greater than 1 (because of the advantage normalization)
        See https://github.com/pytorch/pytorch/issues/29372
    :param batch_size: Minibatch size
    :param n_epochs: Number of epoch when optimizing the surrogate loss
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    :param clip_range: Clipping parameter, it can be a function of the current progress
        remaining (from 1 to 0).
    :param clip_range_vf: Clipping parameter for the value function,
        it can be a function of the current progress remaining (from 1 to 0).
        This is a parameter specific to the OpenAI implementation. If None is passed (default),
        no clipping will be done on the value function.
        IMPORTANT: this clipping depends on the reward scaling.
    :param normalize_advantage: Whether to normalize or not the advantage
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param target_kl: Limit the KL divergence between updates,
        because the clipping is not enough to prevent large update
        see issue #213 (cf https://github.com/hill-a/stable-baselines/issues/213)
        By default, there is no limit on the kl div.
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    policy_aliases: Dict[str, Type[HPPOPolicy]] = {  # type: ignore[assignment]
        "MlpPolicy": HPPOPolicy,
        # "CnnPolicy": ActorCriticCnnPolicy,
        # "MultiInputPolicy": MultiInputActorCriticPolicy,
    }
    policy: HPPOPolicy  # type: ignore[assignment]

    def __init__(
        self,
        policy: Union[str, Type[HPPOPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        target_kl: Optional[float] = None,
        support_multi_env: bool = True,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = {"net_arch":{"n_units":64,"n_options":2}},
        env_kwargs: Optional[Dict[str, Any]] = {"num_envs": 1},
        verbose: int = 0,
        seed: Optional[int] = None,
        device: str = "auto",
        _init_setup_model: bool = True,
    ):
        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            # Note: gSDE is not properly implemented,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            support_multi_env=support_multi_env,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            env_kwargs=env_kwargs,
            verbose=verbose,
            device=device,
            seed=seed,
            _init_setup_model=_init_setup_model,
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                # spaces.MultiDiscrete,
                # spaces.MultiBinary,
            ),
        )

        # Sanity check, otherwise it will lead to noisy gradient and NaN
        # because of the advantage normalization
        if normalize_advantage:
            assert (
                batch_size > 1
            ), "`batch_size` must be greater than 1. See https://github.com/DLR-RM/stable-baselines3/issues/440"

        if self.env is not None:
            # Check that `n_steps * n_envs > 1` to avoid NaN
            # when doing advantage normalization
            buffer_size = self.env.num_envs * self.n_steps
            assert buffer_size > 1 or (
                not normalize_advantage
            ), f"`n_steps * n_envs` must be greater than 1. Currently n_steps={self.n_steps} and n_envs={self.env.num_envs}"
            # Check that the rollout buffer size is a multiple of the mini-batch size
            untruncated_batches = buffer_size // batch_size
            if buffer_size % batch_size > 0:
                warnings.warn(
                    f"You have specified a mini-batch size of {batch_size},"
                    f" but because the `RolloutBuffer` is of size `n_steps * n_envs = {buffer_size}`,"
                    f" after every {untruncated_batches} untruncated mini-batches,"
                    f" there will be a truncated mini-batch of size {buffer_size % batch_size}\n"
                    f"We recommend using a `batch_size` that is a factor of `n_steps * n_envs`.\n"
                    f"Info: (n_steps={self.n_steps} and n_envs={self.env.num_envs})"
                )

        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.clip_range_schedule = get_schedule_fn(self.clip_range)
        self.normalize_advantage = normalize_advantage
        self.target_kl = target_kl

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()

        if not hasattr(self, "policy") or self.policy is None:  # type: ignore[has-type]
            # pytype:disable=not-instantiable
            self.policy = self.policy_class(  # type: ignore[assignment]
                self.observation_space,
                self.action_space,
                self.lr_schedule,
                **self.policy_kwargs,
            )
            # pytype:enable=not-instantiable

            self.key = self.policy.build(self.key, self.lr_schedule, self.max_grad_norm)

            self.key, ent_key = jax.random.split(self.key, 2)


        # Initialize schedules for policy/value clipping
    #    self.clip_range_schedule = get_schedule_fn(self.clip_range)
        # if self.clip_range_vf is not None:
        #     if isinstance(self.clip_range_vf, (float, int)):
        #         assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, " "pass `None` to deactivate vf clipping"
        #
        #     self.clip_range_vf = get_schedule_fn(self.clip_range_vf)

    @staticmethod
    @partial(jax.jit, static_argnames=["normalize_advantage"])
    def _one_update(
        actor_state: TrainState,
        option_actor_state: TrainState,
        option_starter_state: TrainState,
        vf_state: TrainState,
        option_value_state: TrainState,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        options: jnp.ndarray,
        option_starts: jnp.ndarray,
        episode_starts: jnp.ndarray,
        last_options: jnp.ndarray,
        advantages: jnp.ndarray,
        option_advantages: jnp.ndarray,
        option_start_advantages: jnp.ndarray,
        returns: jnp.ndarray,
        old_log_prob: jnp.ndarray,
        old_option_log_probs: jnp.ndarray,
        option_start_log_probs: jnp.ndarray,
        clip_range: float,
        ent_coef: float,
        vf_coef: float,
        normalize_advantage: bool = True,
    ):
        # Normalize advantage
        # Normalization does not make sense if mini batchsize == 1, see GH issue #325
        if normalize_advantage and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            option_advantages = (option_advantages - option_advantages.mean()) / (option_advantages.std() + 1e-8)
            option_start_advantages = (option_start_advantages - option_start_advantages.mean()) / (option_start_advantages.std() + 1e-8)

        def option_actor_loss(params):
            dist = option_actor_state.apply_fn(params, observations)
            option_log_probs = dist.log_prob(options)
            entropy = dist.entropy()

            old_option_log_prob = jnp.where( option_starts,  old_option_log_probs,  option_start_log_probs)
            # ratio between old and new policy, should be one at the first iteration
            ratio = jnp.exp(option_log_probs - old_option_log_prob)
            # clipped surrogate loss
            policy_loss_1 = option_advantages * ratio
            policy_loss_2 = option_advantages * jnp.clip(ratio, 1 - clip_range, 1 + clip_range)
            policy_loss = -jnp.minimum(policy_loss_1, policy_loss_2).mean()

            # Entropy loss favor exploration
            # Approximate entropy when no analytical form
            # entropy_loss = -jnp.mean(-log_prob)
            # analytical form
            entropy_loss = -jnp.mean(entropy)

            total_option_policy_loss = policy_loss + ent_coef * entropy_loss
            return total_option_policy_loss

        def option_start_loss(params):
            dist = option_starter_state.apply_fn(params, observations,last_options)
            log_prob = dist.log_prob(actions)
            entropy = dist.entropy()

            # ratio between old and new policy, should be one at the first iteration
            ratio = jnp.exp(log_prob - old_log_prob)
            # clipped surrogate loss
            true_option_start_advantages = option_start_advantages * (1-episode_starts)
            policy_loss_1 = true_option_start_advantages * ratio
            policy_loss_2 = true_option_start_advantages * jnp.clip(ratio, 1 - clip_range, 1 + clip_range)
            policy_loss = -jnp.minimum(policy_loss_1, policy_loss_2).mean()

            # Entropy loss favor exploration
            # Approximate entropy when no analytical form
            # entropy_loss = -jnp.mean(-log_prob)
            # analytical form
            entropy_loss = -jnp.mean(entropy)

            total_policy_loss = policy_loss + ent_coef * entropy_loss
            return total_policy_loss

        def actor_loss(params):
            dist = actor_state.apply_fn(params, observations,options)
            log_prob = dist.log_prob(actions)
            entropy = dist.entropy()

            # ratio between old and new policy, should be one at the first iteration
            ratio = jnp.exp(log_prob - old_log_prob)
            # clipped surrogate loss
            policy_loss_1 = advantages * ratio
            policy_loss_2 = advantages * jnp.clip(ratio, 1 - clip_range, 1 + clip_range)
            policy_loss = -jnp.minimum(policy_loss_1, policy_loss_2).mean()

            # Entropy loss favor exploration
            # Approximate entropy when no analytical form
            # entropy_loss = -jnp.mean(-log_prob)
            # analytical form
            entropy_loss = -jnp.mean(entropy)

            total_policy_loss = policy_loss + ent_coef * entropy_loss
            return total_policy_loss

        pg_loss_value, grads = jax.value_and_grad(actor_loss, has_aux=False)(actor_state.params)
        actor_state = actor_state.apply_gradients(grads=grads)

        option_loss_value, grads = jax.value_and_grad(option_actor_loss, has_aux=False)(option_actor_state.params)
        option_actor_state = option_actor_state.apply_gradients(grads=grads)

        option_start_loss_value, grads = jax.value_and_grad(option_actor_loss, has_aux=False)(option_starter_state.params)
        option_starter_state = option_starter_state.apply_gradients(grads=grads)

        def critic_loss(params):
            # Value loss using the TD(gae_lambda) target
            vf_values = vf_state.apply_fn(params, observations).flatten()
            return ((returns - vf_values) ** 2).mean()

        def option_critic_loss(params):
            # Value loss using the TD(gae_lambda) target
            option_values = option_value_state.apply_fn(params, observations,options).flatten()
            return ((returns - option_values) ** 2).mean()
        vf_loss_value, grads = jax.value_and_grad(critic_loss, has_aux=False)(vf_state.params)
        vf_state = vf_state.apply_gradients(grads=grads)

        option_loss_value, grads = jax.value_and_grad(option_critic_loss, has_aux=False)(option_value_state.params)
        option_value_state = option_value_state.apply_gradients(grads=grads)

        # loss = policy_loss + ent_coef * entropy_loss + vf_coef * value_loss
        return (actor_state, option_actor_state, option_starter_state,vf_state,option_value_state), \
               (pg_loss_value, option_loss_value, option_start_loss_value,vf_loss_value,option_loss_value)

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Update optimizer learning rate
        # self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range_schedule(self._current_progress_remaining)

        # train for n_epochs epochs
        for _ in range(self.n_epochs):
            # JIT only one update
            self.key, random_key = jax.random.split(self.key, 2)
            for rollout_data in self.rollout_buffer.get(random_key,self.batch_size):  # type: ignore[attr-defined]
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to int
                    actions = rollout_data.actions.flatten().astype(jnp.int32)
                else:
                    actions = rollout_data.actions

                (self.policy.actor_state, self.policy.option_actor_state,self.policy.option_starter_state,
                  self.policy.vf_state,self.policy.option_value_state ), \
                (pg_loss_value, option_loss_value, option_start_loss_value,vf_loss_value,option_loss_value) = self._one_update(
                    actor_state=self.policy.actor_state,
                option_actor_state = self.policy.option_actor_state,
                option_starter_state= self.policy.option_starter_state,
                vf_state=self.policy.vf_state,
                option_value_state=self.policy.option_value_state,
                    observations=rollout_data.observations,
                    actions=actions,
                options= rollout_data.options,
                option_starts= rollout_data.option_starts,
                episode_starts=rollout_data.episode_starts,
                    last_options=rollout_data.last_options,
                    advantages=rollout_data.advantages,
                    option_advantages=rollout_data.option_advantages,
                    option_start_advantages=rollout_data.option_start_advantages,
                    returns=rollout_data.returns,
                    old_log_prob=rollout_data.old_log_prob,
                    old_option_log_probs=rollout_data.old_option_log_probs,
                    option_start_log_probs=rollout_data.option_start_log_probs,
                    clip_range=clip_range,
                    ent_coef=self.ent_coef,
                    vf_coef=self.vf_coef,
                    normalize_advantage=self.normalize_advantage,
                )

        self._n_updates += self.n_epochs
        explained_var = explained_variance(
            jnp.concatenate(self.rollout_buffer.values).flatten(),  # type: ignore[attr-defined]
            jnp.concatenate(self.rollout_buffer.returns).flatten(),  # type: ignore[attr-defined]
        )

        # Logs
        # self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        # self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        # TODO: use mean instead of one point
        self.logger.record("train/value_loss", vf_loss_value.item())
        # self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        # self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/pg_loss", pg_loss_value.item())
        self.logger.record("train/explained_variance", explained_var)
        # if hasattr(self.policy, "log_std"):
        #     self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        # if self.clip_range_vf is not None:
        #     self.logger.record("train/clip_range_vf", clip_range_vf)

    def learn(
        self: HPPOSelf,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "HPPO",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> HPPOSelf:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )
