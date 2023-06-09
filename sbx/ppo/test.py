
from sbx import  PPO

from sbx.common.evaluation import evaluate_policy
model = PPO("MlpPolicy", "Pendulum-v1", learning_rate=1e-3,n_steps=1024,batch_size=64,env_kwargs={"num_envs":4},policy_kwargs={"net_arch":(64,)}, verbose=1)
model.learn(total_timesteps= 100000, progress_bar=True)

vec_env = model.get_env()

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