from typing import Dict, List, Tuple, Type, Union, Callable

import jax.numpy as jnp
from gymnax.environments import spaces
import flax.linen as nn

from sbx.common.preprocessing import get_flattened_obs_dim, is_image_space
from sbx.common.type_aliases import JnpDict

class BaseFeaturesExtractor(nn.Module):
    """
    Base class that represents a features extractor.

    :param observation_space:
    :param features_dim: Number of features extracted.
    """
    observation_space: spaces.Space
    features_dim: int
    activation_fn: Callable = nn.tanh
class FlattenExtractor(BaseFeaturesExtractor):
    """
    Feature extract that flatten the input.
    Used as a placeholder when feature extraction is not needed.

    :param observation_space:
    """


    def __call__(self, observations: jnp.ndarray) -> jnp.ndarray:
        jnp.reshape(observations, (observations.shape[0], -1))
        return jnp.reshape(observations, (observations.shape[0], -1))



class NatureCNN(BaseFeaturesExtractor):
    """
    CNN from DQN Nature paper:
        Mnih, Volodymyr, et al.
        "Human-level control through deep reinforcement learning."
        Nature 518.7540 (2015): 529-533.

    :param observation_space:
    :param features_dim: Number of features extracted.
        This corresponds to the number of unit for the last layer.
    :param normalized_image: Whether to assume that the image is already normalized
        or not (this disables dtype and bounds checks): when True, it only checks that
        the space is a Box and has 3 dimensions.
        Otherwise, it checks that it has expected dtype (uint8) and bounds (values in [0, 255]).
    """
    normalized_image : bool = False
    def setup(self) -> None:
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        assert is_image_space(self.observation_space, check_channels=False, normalized_image=self.normalized_image), (
            "You should use NatureCNN "
            f"only with images not with {self.observation_space}\n"
            "(you are probably using `CnnPolicy` instead of `MlpPolicy` or `MultiInputPolicy`)\n"
            "If you are using a custom environment,\n"
            "please check it using our env checker:\n"
            "https://stable-baselines3.readthedocs.io/en/master/common/env_checker.html.\n"
            "If you are using `VecNormalize` or already normalized channel-first images "
            "you should pass `normalize_images=False`: \n"
            "https://stable-baselines3.readthedocs.io/en/master/guide/custom_env.html"
        )
        n_input_channels = self.observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv(n_input_channels, 32, kernel_size=(8,8), strides=4),
            self.activation_fn(),
            nn.Conv(32, 64, kernel_size=(4,4), strides=2),
            self.activation_fn(),
            nn.Conv(64, 64, kernel_size=(3,3), strides=1),
            self.activation_fn(),
            FlattenExtractor(),
        )

        # Compute shape by doing one forward pass
        cnn_output_dim = self.cnn(self.observation_space.sample()).shape[1]
        self.linear = nn.Sequential(nn.Dense(cnn_output_dim, self.features_dim), self.activation_fn())

    def __call__(self, observations: jnp.ndarray) -> jnp.ndarray:
        return self.linear(self.cnn(observations))



class CombinedExtractor(BaseFeaturesExtractor):
    """
    Combined features extractor for Dict observation spaces.
    Builds a features extractor for each key of the space. Input from each space
    is fed through a separate submodule (CNN or MLP, depending on input shape),
    the output features are concatenated and fed through additional MLP network ("combined").

    :param observation_space:
    :param cnn_output_dim: Number of features to output from each CNN submodule(s). Defaults to
        256 to avoid exploding network sizes.
    :param normalized_image: Whether to assume that the image is already normalized
        or not (this disables dtype and bounds checks): when True, it only checks that
        the space is a Box and has 3 dimensions.
        Otherwise, it checks that it has expected dtype (uint8) and bounds (values in [0, 255]).
    """

    normalized_image: bool = False,
    cnn_output_dim : int = 256,
    def setup( self) -> None:
        self.extractors = [  NatureCNN(subspace, features_dim=self.cnn_output_dim, normalized_image=self.normalized_image)
                       if is_image_space(subspace, normalized_image=self.normalized_image)
                        else FlattenExtractor(subspace, features_dim=get_flattened_obs_dim(subspace))
                             for key, subspace in self.observation_space.spaces.items() ]


        # Update the features dim manually
        self.features_dim = sum([extractor.features_dim for extractor in self.extractors])

    def __call__(self, observations: JnpDict) -> jnp.ndarray:
        encoded_tensor_list = []
        for extractor in self.extractors:
            encoded_tensor_list.append(extractor(observations))
        return jnp.concatenate(encoded_tensor_list, axis=1)

def get_actor_critic_arch(net_arch: Union[List[int], Dict[str, List[int]]]) -> Tuple[List[int], List[int]]:
    """
    Get the actor and critic network architectures for off-policy actor-critic algorithms (SAC, TD3, DDPG).

    The ``net_arch`` parameter allows to specify the amount and size of the hidden layers,
    which can be different for the actor and the critic.
    It is assumed to be a list of ints or a dict.

    1. If it is a list, actor and critic networks will have the same architecture.
        The architecture is represented by a list of integers (of arbitrary length (zero allowed))
        each specifying the number of units per layer.
       If the number of ints is zero, the network will be linear.
    2. If it is a dict,  it should have the following structure:
       ``dict(qf=[<critic network architecture>], pi=[<actor network architecture>])``.
       where the network architecture is a list as described in 1.

    For example, to have actor and critic that share the same network architecture,
    you only need to specify ``net_arch=[256, 256]`` (here, two hidden layers of 256 units each).

    If you want a different architecture for the actor and the critic,
    then you can specify ``net_arch=dict(qf=[400, 300], pi=[64, 64])``.

    .. note::
        Compared to their on-policy counterparts, no shared layers (other than the features extractor)
        between the actor and the critic are allowed (to prevent issues with target networks).

    :param net_arch: The specification of the actor and critic networks.
        See above for details on its formatting.
    :return: The network architectures for the actor and the critic
    """
    if isinstance(net_arch, list):
        actor_arch, critic_arch = net_arch, net_arch
    else:
        assert isinstance(net_arch, dict), "Error: the net_arch can only contain be a list of ints or a dict"
        assert "pi" in net_arch, "Error: no key 'pi' was provided in net_arch for the actor network"
        assert "qf" in net_arch, "Error: no key 'qf' was provided in net_arch for the critic network"
        actor_arch, critic_arch = net_arch["pi"], net_arch["qf"]
    return actor_arch, critic_arch
