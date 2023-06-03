# import copy

import jax
import jax.numpy as jnp
from gymnax.environments import spaces
from stable_baselines3.common.policies import BasePolicy
from sbx.common.preprocessing import is_image_space, maybe_transpose,preprocess_obs
from sbx.common.utils import is_vectorized_observation

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
        super().__init__()

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



class HierarchicalBaseJaxPolicy(BaseJaxPolicy):

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
    def option_start(self, observations: jnp.ndarray, options: jnp.ndarray,key: jax.random.KeyArray) \
            -> [jnp.ndarray,jnp.ndarray,jnp.ndarray]:
        pass
    def value_function(self, observations: jnp.ndarray) -> jnp.ndarray:
        pass
    def option_value_function(self, observations: jnp.ndarray, options: jnp.ndarray) -> jnp.ndarray:
        pass
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
    def select_action(actor_state, observations):
        return actor_state.apply_fn(actor_state.params, observations).mode()
