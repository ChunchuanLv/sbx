
from sbx import  HPPO

from sbx.common.evaluation import evaluate_policy
model = HPPO("MlpPolicy", "Pendulum-v1",env_kwargs={"num_envs":4}, verbose=1)
model.learn(total_timesteps=2500, progress_bar=True)

vec_env = model.get_env()

mean_reward, std_reward = evaluate_policy(
    model,
    vec_env,
    n_eval_episodes= 100,
    deterministic= True,
    render = False,
  #  callback: Optional[Callable[[Dict[str, Any], Dict[str, Any]], None]] = None,
   # reward_threshold: Optional[float] = None,
    return_episode_rewards = True,
    warn = True,
)

print (mean_reward, std_reward)