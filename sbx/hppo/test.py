#from jax.config import config
#config.update('jax_disable_jit', True)
from sbx import  HPPO

from sbx.common.evaluation import evaluate_policy
model = HPPO("MlpPolicy", "Pendulum-v1", n_steps=1024,batch_size=64,clip_range=0.5,n_epochs=10,learning_rate=1e-3,env_kwargs={"num_envs":4}, policy_kwargs={"net_arch":{"n_units":64,"n_options":2}},verbose=1)
model.learn(total_timesteps=1e5, progress_bar=True)

vec_env = model.get_env()
#from jax.config import config
#config.update('jax_disable_jit', True)
mean_reward, std_reward = evaluate_policy(
    model,
    vec_env,
    n_eval_episodes= 10,
    deterministic= True,
    render = False,
  #  callback: Optional[Callable[[Dict[str, Any], Dict[str, Any]], None]] = None,
   # reward_threshold: Optional[float] = None,
    return_episode_rewards = False,
    warn = True,
)

print (mean_reward, std_reward)