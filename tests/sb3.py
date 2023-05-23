import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Parallel environments
vec_env = make_vec_env("CartPole-v1", n_envs=2)

model = PPO("MlpPolicy", vec_env, verbose=1)
model.learn(total_timesteps=25)
obs = vec_env.reset()
for i in range(10):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    print(obs, rewards, dones, info)