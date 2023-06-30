# import copy

from functools import partial
import jax
import jax.numpy as jnp
from gymnax.environments import spaces
from stable_baselines3.common.policies import BasePolicy
from sbx.common.preprocessing import is_image_space, maybe_transpose,preprocess_obs
from sbx.common.utils import is_vectorized_observation
from flax.training.train_state import TrainState
import flax.linen as nn
import optax
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union, no_type_check,Callable

from sbx.common.jax_layers import BaseFeaturesExtractor, FlattenExtractor, NatureCNN, CombinedExtractor
class BaseJaxPolicy(BasePolicy):
    features_extractor: nn.Module

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        features_extractor: Optional[BaseFeaturesExtractor] = None,
        normalize_images: bool = True,
        optimizer_class: Callable[..., optax.GradientTransformation] = optax.adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):

        if optimizer_kwargs is None:
            optimizer_kwargs = {}

        if features_extractor_kwargs is None:
            features_extractor_kwargs = {}

        self.observation_space = observation_space
        self.action_space = action_space
        self.features_extractor = features_extractor
        self.normalize_images = normalize_images

        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs

        self.features_extractor_class = features_extractor_class
        self.features_extractor_kwargs = features_extractor_kwargs
        # Automatically deactivate dtype and bounds checks
        if normalize_images is False and issubclass(features_extractor_class, (NatureCNN, CombinedExtractor)):
            self.features_extractor_kwargs.update(dict(normalized_image=True))

        self._squash_output = squash_output

    @staticmethod
    def _dummy_schedule(progress_remaining: float) -> float:
        """(float) Useful for pickling policy."""
        del progress_remaining
        return 0.0

    @property
    def squash_output(self) -> bool:
        """(bool) Getter for squash_output."""
        return self._squash_output

    def scale_action(self, action: jnp.ndarray) -> jnp.ndarray:
        """
        Rescale the action from [low, high] to [-1, 1]
        (no need for symmetric action space)

        :param action: Action to scale
        :return: Scaled action
        """
        low, high = self.action_space.low, self.action_space.high
        return 2.0 * ((action - low) / (high - low)) - 1.0

    def unscale_action(self, scaled_action: jnp.ndarray) -> jnp.ndarray:
        """
        Rescale the action from [-1, 1] to [low, high]
        (no need for symmetric action space)

        :param scaled_action: Action to un-scale
        """
        low, high = self.action_space.low, self.action_space.high
        return low + (0.5 * (scaled_action + 1.0) * (high - low))

    def _update_features_extractor(
        self,
        net_kwargs: Dict[str, Any],
        features_extractor: Optional[BaseFeaturesExtractor] = None,
    ) -> Dict[str, Any]:
        """
        Update the network keyword arguments and create a new features extractor object if needed.
        If a ``features_extractor`` object is passed, then it will be shared.

        :param net_kwargs: the base network keyword arguments, without the ones
            related to features extractor
        :param features_extractor: a features extractor object.
            If None, a new object will be created.
        :return: The updated keyword arguments
        """
        net_kwargs = net_kwargs.copy()
        if features_extractor is None:
            # The features extractor is not shared, create a new one
            features_extractor = self.make_features_extractor()
        net_kwargs.update(dict(features_extractor=features_extractor, features_dim=features_extractor.features_dim))
        return net_kwargs

    def make_features_extractor(self) -> BaseFeaturesExtractor:
        """Helper method to create a features extractor."""
        return self.features_extractor_class(self.observation_space, **self.features_extractor_kwargs)

    def extract_features(self, obs:  jnp.ndarray, features_extractor: BaseFeaturesExtractor) ->  jnp.ndarray:
        """
        Preprocess the observation if needed and extract features.

         :param obs: The observation
         :param features_extractor: The features extractor to use.
         :return: The extracted features
        """
        preprocessed_obs = preprocess_obs(obs, self.observation_space, normalize_images=self.normalize_images)
        return features_extractor(preprocessed_obs)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        """
        Get data that need to be saved in order to re-create the model when loading it from disk.

        :return: The dictionary to pass to the as kwargs constructor when reconstruction this model.
        """
        return dict(
            observation_space=self.observation_space,
            action_space=self.action_space,
            # Passed to the constructor by child class
            # squash_output=self.squash_output,
            # features_extractor=self.features_extractor
            normalize_images=self.normalize_images,
        )


    @classmethod
    def load(cls: Type, path: str, device = "auto"):
        """
        Load model from path.

        :param path:
        :param device: Device on which the policy should be loaded.
        :return:
        TODO
        """
        pass

    def load_from_vector(self, vector: jnp.ndarray) -> None:
        """
        Load parameters from a 1D vector.

        :param vector:
        TODO
        """
        pass

    def parameters_to_vector(self) -> jnp.ndarray:
        """
        Convert the parameters to a 1D vector.

        :return:
        return th.nn.utils.parameters_to_vector(self.parameters()).detach().cpu().numpy()
        TODO
        """
        pass

    def set_training_mode(self, mode: bool) -> None:
        """
        Put the policy in either training or evaluation mode.

        This affects certain modules, such as batch normalisation and dropout.

        :param mode: if true, set to training mode, else set to evaluation mode
        """
        self.train(mode)

    def is_vectorized_observation(self, observation: Union[jnp.ndarray, Dict[str, jnp.ndarray]]) -> bool:
        """
        Check whether or not the observation is vectorized,
        apply transposition to image (so that they are channel-first) if needed.
        This is used in DQN when sampling random action (epsilon-greedy policy)

        :param observation: the input observation to check
        :return: whether the given observation is vectorized or not
        """
        vectorized_env = False
        if isinstance(observation, dict):
            for key, obs in observation.items():
                obs_space = self.observation_space.spaces[key]
                vectorized_env = vectorized_env or is_vectorized_observation(maybe_transpose(obs, obs_space), obs_space)
        else:
            vectorized_env = is_vectorized_observation(
                maybe_transpose(observation, self.observation_space), self.observation_space
            )
        return vectorized_env


    @staticmethod
    @jax.jit
    def sample_action(actor_state, observations, key):
        dist = actor_state.apply_fn(actor_state.params, observations)
        action = dist.sample(seed=key)
        return action

    @staticmethod
    @jax.jit
    def select_action(actor_state, observations):
        return actor_state.apply_fn(actor_state.params, observations).mode()

    @no_type_check
    def predict(
        self,
        observation: Union[jnp.ndarray, Dict[str, jnp.ndarray]],
        state: Optional[Tuple[jnp.ndarray, ...]] = None,
        episode_start: Optional[jnp.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[jnp.ndarray, Optional[Tuple[jnp.ndarray, ...]]]:
        # self.set_training_mode(False)

        observation, vectorized_env = self.prepare_obs(observation)

        actions = self._predict(observation, deterministic=deterministic)

        # Convert to numpy, and reshape to the original action shape
        actions = jnp.array(actions).reshape((-1, *self.action_space.shape))

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

        return actions, state

    def prepare_obs(self, observation: Union[jnp.ndarray, Dict[str, jnp.ndarray]]) -> Tuple[jnp.ndarray, bool]:
        vectorized_env = False
        if isinstance(observation, dict):
            assert isinstance(self.observation_space, spaces.Dict)
            # Minimal dict support: flatten
            keys = list(self.observation_space.keys())
            vectorized_env = is_vectorized_observation(observation[keys[0]], self.observation_space[keys[0]])

            # Add batch dim and concatenate
            observation = jnp.concatenate(
                [observation[key].reshape(-1, *self.observation_space[key].shape) for key in keys],
                axis=1,
            )
            # need to copy the dict as the dict in VecFrameStack will become a torch tensor
            # observation = copy.deepcopy(observation)
            # for key, obs in observation.items():
            #     obs_space = self.observation_space.spaces[key]
            #     if is_image_space(obs_space):
            #         obs_ = maybe_transpose(obs, obs_space)
            #     else:
            #         obs_ = jnp.array(obs)
            #     vectorized_env = vectorized_env or is_vectorized_observation(obs_, obs_space)
            #     # Add batch dimension if needed
            #     observation[key] = obs_.reshape((-1, *self.observation_space[key].shape))

        elif is_image_space(self.observation_space):
            # Handle the different cases for images
            # as PyTorch use channel first format
            observation = maybe_transpose(observation, self.observation_space)

        else:
            vectorized_env = True
            observation = observation

        if not isinstance(self.observation_space, spaces.Dict):
            assert isinstance(observation, jnp.ndarray)
            vectorized_env = is_vectorized_observation(observation, self.observation_space)
            # Add batch dimension if needed
            observation = observation.reshape((-1, *self.observation_space.shape))  # type: ignore[misc]

        assert isinstance(observation, jnp.ndarray)
        return observation, vectorized_env

    def set_training_mode(self, mode: bool) -> None:
        # self.actor.set_training_mode(mode)
        # self.critic.set_training_mode(mode)
        self.training = mode


import tensorflow_probability
tfp = tensorflow_probability.substrates.jax
tfd = tfp.distributions


class HierarchicalBaseJaxPolicy(BaseJaxPolicy):
    actor_state: TrainState
    value_state : TrainState
    option_starter_state: TrainState
    option_actor_state: TrainState
    option_value_state : TrainState

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        features_extractor: Optional[BaseFeaturesExtractor] = None,
        normalize_images: bool = True,
        optimizer_class: Callable[..., optax.GradientTransformation] = optax.adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(observation_space,action_space, squash_output, features_extractor_class,
                            features_extractor_kwargs, features_extractor, normalize_images, optimizer_class,
                            optimizer_kwargs)
    @staticmethod
    @jax.jit
    def total_value(option_start_times: jnp.ndarray,current_times: jnp.ndarray,option_start_log_probs: jnp.ndarray,
                    values: jnp.ndarray,variational_posterior_probs: jnp.ndarray,
                    option_values: jnp.ndarray,entropy: jnp.ndarray,control_values: jnp.ndarray,gamma : float):
        option_start_probs = jnp.exp(option_start_log_probs)
        total_values = option_start_probs * ( values + gamma ** ( option_start_times-current_times) * variational_posterior_probs ) \
                       + (1-option_start_probs) * (option_values -entropy - (1-  gamma ** ( option_start_times-current_times) )* control_values)
        return total_values
    @staticmethod
    @jax.jit
    def _value_function(value_state, observations: jnp.ndarray) -> jnp.ndarray:
        return value_state.apply_fn(value_state.params, observations).flatten()
    def value_function(self, observations: jnp.ndarray) -> jnp.ndarray:
        return self._value_function(self.value_state, observations)
    @staticmethod
    @jax.jit
    def _option_value_function(option_value_state, observations: jnp.ndarray, options: jnp.ndarray) -> jnp.ndarray:
        return option_value_state.apply_fn(option_value_state.params, observations, options).flatten()
    def option_value_function(self, observations: jnp.ndarray, options: jnp.ndarray) -> jnp.ndarray:
        return self._option_value_function(self.option_value_state, observations, options)
    @staticmethod
    @jax.jit
    def sample_option(option_actor_state, observations, key):
        dist = option_actor_state.apply_fn(option_actor_state.params, observations)
        option = dist.sample(seed=key)
        return option

    @staticmethod
    @jax.jit
    def select_option(option_actor_state, observations):
        dist = option_actor_state.apply_fn(option_actor_state.params, observations)
        return dist.mode()

    @staticmethod
    @jax.jit
    def sample_option_starter(option_starter_state, observations, old_options, key):
        dist = option_starter_state.apply_fn(option_starter_state.params, observations,old_options)
        switch = dist.sample(seed=key)
        return switch

    @staticmethod
    @jax.jit
    def select_option_starter(option_starter_state, observations, old_options):
        dist = option_starter_state.apply_fn(option_starter_state.params, observations,old_options)
        return dist.mode()

    @staticmethod
    @jax.jit
    def sample_action(actor_state, observations, options, key):
        dist = actor_state.apply_fn(actor_state.params, observations,options)
        action = dist.sample(seed=key)
        return action

    @staticmethod
    @jax.jit
    def select_action(actor_state, observations,options):
        return actor_state.apply_fn(actor_state.params, observations,options).mode()

    @staticmethod
    @jax.jit
    def _option_start(option_starter_state, observations: jnp.ndarray, options: jnp.ndarray,dones:jnp.ndarray, key: jax.random.KeyArray) \
            -> [jnp.ndarray,jnp.ndarray,jnp.ndarray]:
        dummy_log_prob = jnp.zeros(dones.shape, dtype=float).flatten()
        dist = option_starter_state.apply_fn(option_starter_state.params, observations, options.flatten())
        start = dist.sample(seed=key)
        log_prob = dist.log_prob(start)
        start = jnp.logical_or(dones, start.flatten())
        log_prob = jnp.where(dones, dummy_log_prob, log_prob)
        return start.flatten(),  log_prob.flatten()
    def option_start(self, observations: jnp.ndarray, options: jnp.ndarray,dones:jnp.ndarray, key: jax.random.KeyArray) \
            -> [jnp.ndarray,jnp.ndarray,jnp.ndarray]:
        return self._option_start(self.option_starter_state, observations, options,dones, key)
    @staticmethod
    @jax.jit
    def _predict_all(actor_state, value_state,option_actor_state, option_value_state,
                     observations, option, option_start, key):
        
        
        option_dist = option_actor_state.apply_fn(option_actor_state.params, observations)
        new_option = option_dist.sample(seed=key)
        new_option = jnp.where(option_start, new_option, option)
        option_log_probs = option_dist.log_prob(new_option)

        dist = actor_state.apply_fn(actor_state.params, observations,new_option)
        actions = dist.sample(seed=key)
        log_probs = dist.log_prob(actions)
        values = value_state.apply_fn(value_state.params, observations).flatten()

        option_values= option_value_state.apply_fn(option_value_state.params, observations,new_option).flatten()
        return actions, new_option,  log_probs, option_log_probs, values, option_values

    def predict_all(self, observation: jnp.ndarray, option:jnp.ndarray, option_start:jnp.ndarray, key: jax.random.KeyArray) -> Tuple[jnp.ndarray]:
        return self._predict_all(self.actor_state, self.value_state, self.option_actor_state, self.option_value_state,
                                 observation,  option, option_start, key)

    @staticmethod
    @partial(jax.jit, static_argnames=["deterministic"])
    def _predict(option_starter_state, option_actor_state, actor_state,
                 observation: jnp.ndarray,option: jnp.ndarray,episode_start: jnp.ndarray,
                 noise_key: jax.random.KeyArray = None, deterministic: bool = False) -> [jnp.ndarray,jnp.ndarray]:  # type: ignore[override]
        if deterministic:
            select_new_option = HierarchicalBaseJaxPolicy.select_option_starter(option_starter_state, observation, option).flatten()

            select_new_option = jnp.logical_or(select_new_option, episode_start)
            new_option = HierarchicalBaseJaxPolicy.select_option(option_actor_state, observation).flatten()
         #   print("new_option", new_option.shape)
            options = jnp.where(select_new_option, new_option, option)
        #    print("options", options.shape)

            actions = HierarchicalBaseJaxPolicy.select_action(actor_state, observation, options)

        else:
            # Trick to use gSDE: repeat sampled noise by using the same noise key
            select_new_option = HierarchicalBaseJaxPolicy.sample_option_starter(option_starter_state, observation, option,noise_key)

            select_new_option = jnp.logical_or(select_new_option, episode_start)
            new_option = HierarchicalBaseJaxPolicy.sample_option(option_actor_state, observation,noise_key)
            options = jnp.where(select_new_option, new_option, option)

            actions = HierarchicalBaseJaxPolicy.sample_action(actor_state, observation, options,noise_key)
        return actions,options

    def reset_noise(self) -> None:
        """
        Sample new weights for the exploration matrix, when using gSDE.
        """
        raise NotImplementedError

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

        options = state if state is not None else jnp.zeros(episode_start.shape, dtype=int)

        observation, vectorized_env = self.prepare_obs(observation)
        self.reset_noise()
        actions, options = HierarchicalBaseJaxPolicy._predict(self.option_starter_state, self.option_actor_state, self.actor_state,
                                        observation,options.flatten(),episode_start.flatten(),self.noise_key, deterministic)
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

from typing import Callable


from flax import core
class UnsupervisedHierarchicalBaseJaxPolicy(HierarchicalBaseJaxPolicy):


    control_value_state : TrainState
    option_reward_value_state : TrainState
    variational_posterior_state : TrainState
    #variational_posterior_apply_fn: Callable[[core.FrozenDict[str, Any], jnp.ndarray], tfd.Distribution]
    @staticmethod
    @jax.jit
    def _control_value_function(control_value_state, observations: jnp.ndarray, options: jnp.ndarray) -> jnp.ndarray:
        return control_value_state.apply_fn(control_value_state.params, observations, options).flatten()
    def control_value_function(self, observations: jnp.ndarray, options: jnp.ndarray) -> jnp.ndarray:
        return self._control_value_function(self.control_value_state, observations, options)
    @staticmethod
    @jax.jit
    def _variational_posterior_log_prob(variational_posterior_state,observations: jnp.ndarray, options: jnp.ndarray) -> jnp.ndarray:
        variational_posterior_dist = variational_posterior_state.apply_fn(variational_posterior_state.params, observations)
        variational_posterior_log_probs = variational_posterior_dist.log_prob(options)
        return variational_posterior_log_probs.flatten()
    def variational_posterior_log_prob(self, observations: jnp.ndarray, options: jnp.ndarray) -> jnp.ndarray:
        return self._variational_posterior_log_prob(self.variational_posterior_state, observations, options)
    @staticmethod
    @jax.jit
    def _policy_entropy(option_actor_state,observations: jnp.ndarray) -> jnp.ndarray:

        option_dist = option_actor_state.apply_fn(option_actor_state.params, observations)
        entropy = option_dist.entropy()
        return entropy
    def policy_entropy(self,observations: jnp.ndarray) -> jnp.ndarray:
        return self._policy_entropy(self.option_actor_state, observations)
    @staticmethod
    @partial(jax.jit, static_argnames=["dummy_option"])
  #  @jax.jit
    def _predict_all(actor_state, value_state,option_actor_state, option_value_state,option_reward_value_state,
                     control_value_state,   variational_posterior_state,option_starter_state,
                     observations, old_option, episode_start, key,dummy_option = True):
        dummy_log_prob = jnp.zeros(episode_start.shape, dtype=float).flatten()
   #     print ("observations",observations.shape)
   #     print ("old_option",old_option.shape)
   #     print ("episode_start",episode_start.shape)
        option_start_dist = option_starter_state.apply_fn(option_starter_state.params, observations, old_option.flatten())
        option_start = option_start_dist.sample(seed=key)
        option_start_log_probs =  option_start_dist.log_prob(option_start).flatten()

        option_start = jnp.logical_or(episode_start,  option_start.flatten())
     #   print ("option_start",option_start.shape)
    #    print ("episode_start",episode_start.shape)
        option_start_log_probs = jnp.where(episode_start, dummy_log_prob,option_start_log_probs)


        option_dist = option_actor_state.apply_fn(option_actor_state.params, observations,dummy_option)
        entropy = option_dist.entropy()
        new_option = option_dist.sample(seed=key)
        new_option = jnp.where(option_start, new_option, old_option)
        option_log_probs = option_dist.log_prob(new_option)

        dist = actor_state.apply_fn(actor_state.params, observations,new_option)
        actions = dist.sample(seed=key)
        log_probs = dist.log_prob(actions)
        values = value_state.apply_fn(value_state.params, observations).flatten()

        option_values= option_value_state.apply_fn(option_value_state.params, observations,new_option).flatten()
        last_option_values= option_value_state.apply_fn(option_value_state.params, observations,old_option).flatten()

        option_reward_values = option_reward_value_state.apply_fn(option_reward_value_state.params, observations, new_option).flatten()
        control_values = control_value_state.apply_fn(control_value_state.params, observations,new_option).flatten()
        last_option_control_values = control_value_state.apply_fn(control_value_state.params, observations,old_option).flatten()
        variational_posterior_dist = variational_posterior_state.apply_fn(variational_posterior_state.params, observations)
        variational_posterior_log_probs = variational_posterior_dist.log_prob(old_option).flatten()
        return actions,option_start, new_option,  log_probs, option_log_probs, values, option_values,\
            last_option_values,option_reward_values, last_option_control_values,control_values,entropy, variational_posterior_log_probs,option_start_log_probs

    def predict_all(self, observation: jnp.ndarray, old_option:jnp.ndarray, episode_start:jnp.ndarray, key: jax.random.KeyArray,dummy_option=True) -> Tuple[jnp.ndarray]:
        return self._predict_all(self.actor_state, self.value_state, self.option_actor_state, self.option_value_state,self.option_reward_value_state,
                                 self.control_value_state,self.variational_posterior_state,self.option_starter_state,
                                 observation,  old_option, episode_start, key,dummy_option)