from typing import Any, Callable, Dict, Tuple, Optional, Union

import flax.linen as nn
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow_probability
from flax.linen.initializers import constant
from flax.training.train_state import TrainState
#from gymnasium import spaces
import gymnax.environments.spaces as spaces
from stable_baselines3.common.type_aliases import Schedule

from sbx.common.policies import HierarchicalBaseJaxPolicy

tfp = tensorflow_probability.substrates.jax
tfd = tfp.distributions


class OptionCrtic(nn.Module):
    n_options: int = 16
    n_units: int = 256
    activation_fn: Callable = nn.tanh

    @nn.compact
    def __call__(self, x: jnp.ndarray,z:jnp.ndarray) -> jnp.ndarray:
        z_embed = nn.Embed(self.n_options,self.n_units)(z)
        x = nn.Dense(self.n_units)(x)+ z_embed
        x = self.activation_fn(x)
        x = nn.Dense(self.n_units)(x)
        x = self.activation_fn(x)
        x = nn.Dense(1)(x)
        return x

class OptionStarter(nn.Module):
    n_units: int = 256
    n_options: int = 16
    activation_fn: Callable = nn.tanh

    @nn.compact
    def __call__(self, x: jnp.ndarray,z:jnp.ndarray) -> tfd.Distribution:  # type: ignore[name-defined]
        z_embed = nn.Embed(self.n_options,self.n_units)(z)
        x = nn.Dense(self.n_units)(x)+ z_embed
        x = self.activation_fn(x)
        x = nn.Dense(self.n_units)(x)
        x = self.activation_fn(x)
        mean = nn.Dense(1)(x)
        return tfd.Bernoulli(logits=mean)

class OptionActor(nn.Module):
    n_options: int = 16
    n_units: int = 256
    activation_fn: Callable = nn.tanh

    def get_std(self):
        # Make it work with gSDE
        return jnp.array(0.0)

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> tfd.Distribution:  # type: ignore[name-defined]
        x = nn.Dense(self.n_units)(x)
        x = self.activation_fn(x)
        x = nn.Dense(self.n_units)(x)
        x = self.activation_fn(x)
        mean = nn.Dense(self.n_options)(x)
        dist = tfd.Categorical(logits=mean)
        return dist

class Critic(nn.Module):
    n_units: int = 256
    activation_fn: Callable = nn.tanh

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(self.n_units)(x)
        x = self.activation_fn(x)
        x = nn.Dense(self.n_units)(x)
        x = self.activation_fn(x)
        x = nn.Dense(1)(x)
        return x

class Actor(nn.Module):
    action_dim: int
    n_options: int
    n_units: int = 256
    log_std_init: float = 0.0
    continuous: bool = True
    activation_fn: Callable = nn.tanh

    def get_std(self):
        # Make it work with gSDE
        return jnp.array(0.0)

    @nn.compact
    def __call__(self, x: jnp.ndarray,z:jnp.ndarray) -> tfd.Distribution:  # type: ignore[name-defined]
        z_embed = nn.Embed(self.n_options,self.n_units)(z)
        x = nn.Dense(self.n_units)(x)+ z_embed
        x = self.activation_fn(x)
        x = nn.Dense(self.n_units)(x)
        x = self.activation_fn(x)
        mean = nn.Dense(self.action_dim)(x)
        if self.continuous:
            log_std = self.param("log_std", constant(self.log_std_init), (self.action_dim,))
            dist = tfd.MultivariateNormalDiag(loc=mean, scale_diag=jnp.exp(log_std))
        else:
            dist = tfd.Categorical(logits=mean)
        return dist


class HPPOPolicy(HierarchicalBaseJaxPolicy):
    optimizer_class: Callable[..., optax.GradientTransformation] = optax.adam
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Dict[str, int]] = None,
        ortho_init: bool = False,
        log_std_init: float = 0.0,
        activation_fn=nn.tanh,
        use_sde: bool = False,
        # Note: most gSDE parameters are not used
        # this is to keep API consistent with SB3
        use_expln: bool = False,
        clip_mean: float = 2.0,
        features_extractor_class=None,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Callable[..., optax.GradientTransformation] = optax.adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        share_features_extractor: bool = False,
    ):
        if optimizer_kwargs is None:
            # Small values to avoid NaN in Adam optimizer
            optimizer_kwargs = {}
            if optimizer_class == optax.adam:
                optimizer_kwargs["eps"] = 1e-5

        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=True,
        )
        self.log_std_init = log_std_init
        self.activation_fn = activation_fn
        if net_arch is not None:
            self.n_units = net_arch["n_units"]
            self.n_options = net_arch["n_options"]
        else:
            self.n_units = 64
            self.n_options = 16
        self.use_sde = use_sde

        self.key = self.noise_key = jax.random.PRNGKey(0)

    def build(self, key: jax.random.KeyArray, lr_schedule: Schedule, max_grad_norm: float) -> jax.random.KeyArray:
        key, actor_key, vf_key, option_key,switcher_key, option_value_key = jax.random.split(key, 6)
        # Keep a key for the actor
        key, self.key = jax.random.split(key, 2)
        # Initialize noise
        self.reset_noise()

        obs = jnp.array([self.observation_space.sample(self.noise_key)])

        if isinstance(self.action_space, spaces.Box):
            actor_kwargs = {
                "action_dim": int(np.prod(self.action_space.shape)),
                "continuous": True,
            }
        elif isinstance(self.action_space, spaces.Discrete):
            actor_kwargs = {
                "action_dim": int(self.action_space.n),
                "continuous": False,
            }
        else:
            raise NotImplementedError(f"{self.action_space}")

        self.option_actor = OptionActor(
            n_options=self.n_options,
            n_units=self.n_units,
            log_std_init=self.log_std_init,
            activation_fn=self.activation_fn,
            **actor_kwargs,  # type: ignore[arg-type]
        )
        self.option_actor.reset_noise = self.reset_noise
        self.option_actor_state = TrainState.create(
            apply_fn=self.option_actor.apply,
            params=self.option_actor.init(option_key, obs),
            tx=optax.chain(
                optax.clip_by_global_norm(max_grad_norm),
                self.optimizer_class(
                    learning_rate=lr_schedule(1),  # type: ignore[call-arg]
                    **self.optimizer_kwargs,  # , eps=1e-5
                ),
            ),
        )
        
        self.option_starter = OptionStarter(
            n_options=self.n_options,
            n_units=self.n_units,
            activation_fn=self.activation_fn,
            **actor_kwargs,  # type: ignore[arg-type]
        )
        self.option_starter.reset_noise = self.reset_noise
        self.option_starter_state = TrainState.create(
            apply_fn=self.option_starter.apply,
            params=self.option_starter.init(switcher_key, obs),
            tx=optax.chain(
                optax.clip_by_global_norm(max_grad_norm),
                self.optimizer_class(
                    learning_rate=lr_schedule(1),  # type: ignore[call-arg]
                    **self.optimizer_kwargs,  # , eps=1e-5
                ),
            ),
        )
        
        self.option_value = OptionCrtic(
            n_options=self.n_options,
            n_units=self.n_units,
            activation_fn=self.activation_fn,
            **actor_kwargs,  # type: ignore[arg-type]
        )
        self.option_value_state = TrainState.create(
            apply_fn=self.option_value.apply,
            params=self.option_value.init(option_value_key, obs),
            tx=optax.chain(
                optax.clip_by_global_norm(max_grad_norm),
                self.optimizer_class(
                    learning_rate=lr_schedule(1),  # type: ignore[call-arg]
                    **self.optimizer_kwargs,  # , eps=1e-5
                ),
            ),
        )
        
        self.actor = Actor(
            n_units=self.n_units,
            log_std_init=self.log_std_init,
            activation_fn=self.activation_fn,
            **actor_kwargs,  # type: ignore[arg-type]
        )
        # Hack to make gSDE work without modifying internal SB3 code
        self.actor.reset_noise = self.reset_noise

        self.actor_state = TrainState.create(
            apply_fn=self.actor.apply,
            params=self.actor.init(actor_key, obs),
            tx=optax.chain(
                optax.clip_by_global_norm(max_grad_norm),
                self.optimizer_class(
                    learning_rate=lr_schedule(1),  # type: ignore[call-arg]
                    **self.optimizer_kwargs,  # , eps=1e-5
                ),
            ),
        )

        self.vf = Critic(n_units=self.n_units, activation_fn=self.activation_fn)

        self.value_state = TrainState.create(
            apply_fn=self.vf.apply,
            params=self.vf.init({"params": vf_key}, obs),
            tx=optax.chain(
                optax.clip_by_global_norm(max_grad_norm),
                self.optimizer_class(
                    learning_rate=lr_schedule(1),  # type: ignore[call-arg]
                    **self.optimizer_kwargs,  # , eps=1e-5
                ),
            ),
        )

        self.actor.apply = jax.jit(self.actor.apply)  # type: ignore[method-assign]
        self.vf.apply = jax.jit(self.vf.apply)  # type: ignore[method-assign]
        self.option_value.apply = jax.jit(self.option_value.apply)  # type: ignore[method-assign]
        self.option_starter.apply = jax.jit(self.option_starter.apply)  # type: ignore[method-assign]
        self.option_actor.apply = jax.jit(self.option_actor.apply)  # type: ignore[method-assign]

        return key
    def _sample_new_options(self, observations: jnp.ndarray,options: jnp.ndarray, key: jax.random.KeyArray) -> jnp.ndarray:
        return self.option_actor_state.apply_fn(self.option_actor_state.params, observations,options).sample(seed=key).flatten()
    def option_start(self, observations: jnp.ndarray, options: jnp.ndarray,key: jax.random.KeyArray) \
            -> [jnp.ndarray,jnp.ndarray,jnp.ndarray]:
        dist = self.option_starter_state.apply_fn(self.option_starter_state.params, observations, options)
        start = dist.sample(seed=key)
        return start.flatten(),  dist.log_prob(start).flatten(), dist.logits.flatten()
    def value_function(self, observations: jnp.ndarray) -> jnp.ndarray:
        return self.value_state.apply_fn(self.value_state.params, observations).flatten()
    def option_value_function(self, observations: jnp.ndarray, options: jnp.ndarray) -> jnp.ndarray:
        return self.option_value_state.apply_fn(self.option_value_state.params, observations, options).flatten()
    def reset_noise(self, batch_size: int = 1) -> None:
        """
        Sample new weights for the exploration matrix, when using gSDE.
        """
        self.key, self.noise_key = jax.random.split(self.key, 2)

    def forward(self, obs: jnp.ndarray, deterministic: bool = False) -> jnp.ndarray:
        return self._predict(obs, deterministic=deterministic)


#actions, options,  log_probs, option_log_probs, values, option_values

    def predict_all(self, observation: jnp.ndarray, option:jnp.ndarray, option_start:jnp.ndarray, key: jax.random.KeyArray) -> Tuple[jnp.ndarray]:
        return self._predict_all(self.actor_state, self.value_state, self.option_actor_state, self.option_value_state,
                                 observation,  option, option_start, key)

    @staticmethod
    @jax.jit
    def _predict_all(actor_state, value_state,option_actor_state, option_value_state,
                     obervations, option, option_start, key):

        option_dist = option_actor_state.apply_fn(option_actor_state.params, obervations)
        new_option = option_dist.sample(seed=key)
        new_option = jnp.where(option_start, new_option, option)
        option_log_probs = option_dist.log_prob(new_option)

        dist = actor_state.apply_fn(actor_state.params, obervations,new_option)
        actions = dist.sample(seed=key)
        log_probs = dist.log_prob(actions)
        values = value_state.apply_fn(value_state.params, obervations).flatten()


        option_values= option_value_state.apply_fn(option_value_state.params, obervations,new_option).flatten()
        return actions, new_option,  log_probs, option_log_probs, values, option_values

    @staticmethod
    @jax.jit
    def _predict(option_starter_state, option_actor_state, actor_state,
                 observation: jnp.ndarray,option: jnp.ndarray,episode_start: jnp.ndarray,
                 noise_key: jax.random.KeyArray = None, deterministic: bool = False) -> [jnp.ndarray,jnp.ndarray]:  # type: ignore[override]

        if deterministic:
            select_new_option = HierarchicalBaseJaxPolicy.select_option_starter(option_starter_state, observation, option)

            select_new_option = jnp.logical_or(select_new_option, episode_start)
            new_option = HierarchicalBaseJaxPolicy.select_option(option_actor_state, observation)
            options = jnp.where(select_new_option, new_option, option)

            actions = HierarchicalBaseJaxPolicy.select_action(actor_state, observation, options)

        else:
            # Trick to use gSDE: repeat sampled noise by using the same noise key
            select_new_option = HierarchicalBaseJaxPolicy.sample_option_starter(option_starter_state, observation, option,noise_key)

            select_new_option = jnp.logical_or(select_new_option, episode_start)
            new_option = HierarchicalBaseJaxPolicy.sample_option(option_actor_state, observation,noise_key)
            options = jnp.where(select_new_option, new_option, option)

            actions = HierarchicalBaseJaxPolicy.sample_action(actor_state, observation, options,noise_key)
        return actions,options

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

        option = state if state is not None else jnp.zeros(episode_start.shape, dtype=int)

        observation, vectorized_env = self.prepare_obs(observation)

        self.reset_noise()
        actions, options = HPPOPolicy._predict(self.option_starter_state, self.option_actor_state, self.actor_state,
                                        observation,option,episode_start,self.noise_key, deterministic)
        if isinstance(self.action_space, spaces.Box):
            if self.squash_output:
                # Clip due to numerical instability
                actions = jnp.clip(actions, -1, 1)
                # Rescale to proper domain when using squashing
                actions = self.unscale_action(actions)
            else:
                # Actions could be on arbitrary scale, so clip the actions to avoid
                # out of bound error (e.g. if sampling from a Gaussian distribution)
                actions = jnp.clip(actions, self.action_space.low, self.action_space.high)

        # Remove batch dimension if needed
        if not vectorized_env:
            actions = actions.squeeze(axis=0)  # type: ignore[call-overload]

        return actions, options
