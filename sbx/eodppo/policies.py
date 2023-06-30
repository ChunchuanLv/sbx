from typing import Any, Callable, Dict, Tuple, Optional, Union

from functools import partial
import flax.linen as nn
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.linen.initializers import constant
from flax.training.train_state import TrainState
from stable_baselines3.common.type_aliases import Schedule
from sbx.hppo.policies import HPPOPolicy,OptionCrtic,OptionActor

from sbx.common.policies import UnsupervisedHierarchicalBaseJaxPolicy

import tensorflow_probability
tfp = tensorflow_probability.substrates.jax
tfd = tfp.distributions


class ControlCrtic(nn.Module):
    n_units: int = 256
    n_options: int = 16
    activation_fn: Callable = nn.tanh

    def setup(self):
        self.dense1  = nn.Dense(self.n_units)
        self.dense2 = nn.Dense(self.n_units)
        self.dense3 = nn.Dense(self.n_options)
        self.z_embed = nn.Embed(self.n_options, self.n_units)

    def __call__(self, x: jnp.ndarray, z: jnp.ndarray) -> tfd.Distribution:  # type: ignore[name-defined]
        embed = self.z_embed(z)
        x = self.dense1 (x) + embed
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
        return dist.log_prob(z)

class Variational_Posterior(nn.Module):
    n_units: int = 256
    n_options: int = 16
    activation_fn: Callable = nn.tanh

    def setup(self):
        self.dense1  = nn.Dense(self.n_units)
        self.dense2 = nn.Dense(self.n_units)
        self.dense3 = nn.Dense(self.n_options)

 #   @nn.compact
    def __call__(self, x: jnp.ndarray) -> tfd.Distribution:  # type: ignore[name-defined]
        x = self.dense1(x)
        x = self.activation_fn(x)
        x = self.dense2(x)
        x = self.activation_fn(x)
        x = self.dense3(x)
     #   x = jax.nn.log_softmax(x)
        return tfd.Categorical(logits=x)



class EODPPOPolicy(HPPOPolicy,UnsupervisedHierarchicalBaseJaxPolicy):


    def build(self, key: jax.random.KeyArray, lr_schedule: Schedule, max_grad_norm: float) -> jax.random.KeyArray:
        key = super().build(key, lr_schedule, max_grad_norm)
        key, control_value_key,option_reward_value_key,variational_posterior_key  = jax.random.split(key, 4)

        obs = jnp.array([self.observation_space.sample(self.noise_key)])
        opts = jnp.zeros((obs.shape[0]), dtype=jnp.int32)

        self.control_value = ControlCrtic(
            n_options=self.n_options,
            n_units=self.n_units,
            activation_fn=self.activation_fn,
        )
        self.control_value_state = TrainState.create(
            apply_fn=self.control_value.apply,
            params=self.control_value.init(control_value_key, obs, opts),
            tx=optax.chain(
                optax.clip_by_global_norm(max_grad_norm),
                self.optimizer_class(
                    learning_rate=lr_schedule(1),  # type: ignore[call-arg]
                    **self.optimizer_kwargs,  # , eps=1e-5
                ),
            ),
        )


        self.option_reward_value = ControlCrtic(
            n_options=self.n_options,
            n_units=self.n_units,
            activation_fn=self.activation_fn,
        )
        self.option_reward_value_state = TrainState.create(
            apply_fn=self.option_reward_value.apply,
            params=self.option_reward_value.init(option_reward_value_key, obs, opts),
            tx=optax.chain(
                optax.clip_by_global_norm(max_grad_norm),
                self.optimizer_class(
                    learning_rate=lr_schedule(1),  # type: ignore[call-arg]
                    **self.optimizer_kwargs,  # , eps=1e-5
                ),
            ),
        )


        self.variational_posterior = Variational_Posterior(
            n_options=self.n_options,
            n_units=self.n_units,
            activation_fn=self.activation_fn,
        )

        self.variational_posterior_state = TrainState.create(
            apply_fn=self.variational_posterior.apply,
            params=self.variational_posterior.init(variational_posterior_key, obs),
            tx=optax.chain(
                optax.clip_by_global_norm(max_grad_norm),
                self.optimizer_class(
                    learning_rate=lr_schedule(1),  # type: ignore[call-arg]
                    **self.optimizer_kwargs,  # , eps=1e-5
                ),
            ),
        )

        self.variational_posterior.reset_noise = self.reset_noise


        self.control_value.apply = jax.jit(self.control_value.apply)  # type: ignore[method-assign]
        self.option_reward_value.apply = jax.jit(self.option_reward_value.apply)  # type: ignore[method-assign]
        self.variational_posterior.apply = jax.jit(self.variational_posterior.apply)  # type: ignore[method-assign]
    #    self.variational_posterior_apply_fn = self.variational_posterior.apply
        return key
