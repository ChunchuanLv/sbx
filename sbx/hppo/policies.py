from typing import Any, Callable, Dict, Tuple, Optional, Union

import flax.linen as nn
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.linen.initializers import constant
from flax.training.train_state import TrainState
# from gymnasium import spaces
import gymnax.environments.spaces as spaces
from stable_baselines3.common.type_aliases import Schedule

from sbx.common.policies import HierarchicalBaseJaxPolicy

import tensorflow_probability
tfp = tensorflow_probability.substrates.jax
tfd = tfp.distributions


class OptionCrtic(nn.Module):
    n_options: int = 16
    n_units: int = 256
    activation_fn: Callable = nn.tanh

    @nn.compact
    def __call__(self, x: jnp.ndarray, z: jnp.ndarray) -> jnp.ndarray:
        z_embed = nn.Embed(self.n_options, self.n_units)(z)
        x = nn.Dense(self.n_units)(x) + z_embed
        x = self.activation_fn(x)
        x = nn.Dense(self.n_units)(x)
        x = self.activation_fn(x)
        x = nn.Dense(1)(x)
        return x


class OptionStarter(nn.Module):
    n_units: int = 256
    n_options: int = 16
    activation_fn: Callable = nn.tanh

    def setup(self):
        self.dense1  = nn.Dense(self.n_units)
        self.dense2 = nn.Dense(self.n_units)
        self.dense3 = nn.Dense(self.n_options)

    def __call__(self, x: jnp.ndarray, z: jnp.ndarray) -> tfd.Distribution:  # type: ignore[name-defined]
        x = self.dense1 (x)
        x = self.activation_fn(x)
        x = self.dense2(x)
        x = self.activation_fn(x)
        x = self.dense3(x)
    #    x = jax.nn.log_softmax(x)
        dist = tfd.Categorical(logits=x)
    #    x = jax.nn.softmax(x)
     #   mean_l = jnp.take(x,z)
     #   logits= - jnp.log(jnp.exp(-mean_l)-1)
     #   tfd.Bernoulli(probs=jnp.exp(dist.log_prob(z)))
        return tfd.Bernoulli(probs=jnp.exp(dist.log_prob(z)))


class OptionActor(nn.Module):
    n_options: int = 16
    n_units: int = 256
    activation_fn: Callable = nn.tanh

    def get_std(self):
        # Make it work with gSDE
        return jnp.array(0.0)

    @nn.compact
    def __call__(self, x: jnp.ndarray,dummy_option=False) -> tfd.Distribution:  # type: ignore[name-defined]
        if dummy_option:
            return tfd.Categorical(logits=jnp.zeros( (x.shape[0],self.n_options)))
        x = nn.Dense(self.n_units)(x)
        x = self.activation_fn(x)
        x = nn.Dense(self.n_units)(x)
        x = self.activation_fn(x)
        mean = nn.Dense(self.n_options)(x)
    #    if dummy_option: mean = mean * 0
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
    def __call__(self, x: jnp.ndarray, z: jnp.ndarray) -> tfd.Distribution:  # type: ignore[name-defined]
        #     print ("options",z.shape,z.dtype,z)
        #      print ("obs",x.shape,x.dtype,x)
        z_embed = nn.Embed(self.n_options, self.n_units)(z)
        x = nn.Dense(self.n_units)(x) + z_embed
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
            squash_output=True,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
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
        key, actor_key, value_key, option_key, option_starter_key, option_value_key = jax.random.split(key, 6)
        # Keep a key for the actor
        key, self.key = jax.random.split(key, 2)
        # Initialize noise
        self.reset_noise()

        obs = jnp.array([self.observation_space.sample(self.noise_key)])
        opts = jnp.zeros((obs.shape[0]), dtype=jnp.int32)

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
            activation_fn=self.activation_fn,
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
        )
        self.option_starter.reset_noise = self.reset_noise
        self.option_starter_state = TrainState.create(
            apply_fn=self.option_starter.apply,
            params=self.option_starter.init(option_starter_key, obs, opts),
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
        )
        self.option_value_state = TrainState.create(
            apply_fn=self.option_value.apply,
            params=self.option_value.init(option_value_key, obs, opts),
            tx=optax.chain(
                optax.clip_by_global_norm(max_grad_norm),
                self.optimizer_class(
                    learning_rate=lr_schedule(1),  # type: ignore[call-arg]
                    **self.optimizer_kwargs,  # , eps=1e-5
                ),
            ),
        )

        self.actor = Actor(
            n_options=self.n_options,
            n_units=self.n_units,
            log_std_init=self.log_std_init,
            activation_fn=self.activation_fn,
            **actor_kwargs,  # type: ignore[arg-type]
        )
        # Hack to make gSDE work without modifying internal SB3 code
        self.actor.reset_noise = self.reset_noise

        self.actor_state = TrainState.create(
            apply_fn=self.actor.apply,
            params=self.actor.init(actor_key, obs, opts),
            tx=optax.chain(
                optax.clip_by_global_norm(max_grad_norm),
                self.optimizer_class(
                    learning_rate=lr_schedule(1),  # type: ignore[call-arg]
                    **self.optimizer_kwargs,  # , eps=1e-5
                ),
            ),
        )

        self.value = Critic(n_units=self.n_units, activation_fn=self.activation_fn)

        self.value_state = TrainState.create(
            apply_fn=self.value.apply,
            params=self.value.init({"params": value_key}, obs),
            tx=optax.chain(
                optax.clip_by_global_norm(max_grad_norm),
                self.optimizer_class(
                    learning_rate=lr_schedule(1),  # type: ignore[call-arg]
                    **self.optimizer_kwargs,  # , eps=1e-5
                ),
            ),
        )

        self.actor.apply = jax.jit(self.actor.apply)  # type: ignore[method-assign]
        self.value.apply = jax.jit(self.value.apply)  # type: ignore[method-assign]
        self.option_value.apply = jax.jit(self.option_value.apply)  # type: ignore[method-assign]
        self.option_starter.apply = jax.jit(self.option_starter.apply)  # type: ignore[method-assign]
        self.option_actor.apply = jax.jit(self.option_actor.apply,static_argnames=["dummy_option"])  # type: ignore[method-assign]

        return key

    def reset_noise(self) -> None:
        """
        Sample new weights for the exploration matrix, when using gSDE.
        """
        self.key, self.noise_key = jax.random.split(self.key, 2)
