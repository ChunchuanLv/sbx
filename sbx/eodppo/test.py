from sbx import  EODPPO
import math
from sbx.common.evaluation import evaluate_policy
from bayes_opt import BayesianOptimization
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
import random
env_name = "Pendulum-v1"
def evaluate_hyper_parameters(log_train_freq=0,log_gradient_steps=None,gamma=0.9):

    model = EODPPO("MlpPolicy", env_name, gamma=gamma,seed=random.seed(),clip_range=0.5,
                   n_steps=1024, batch_size=64,  n_epochs=10, learning_rate=1e-3,
                   env_kwargs={"num_envs": 4},policy_kwargs={"net_arch":{"n_units":64,"n_options":4}},verbose=1)
    #Pendulum-v1 -1184 random
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
    if math.isnan(mean_reward.item()): return -2000
    return mean_reward.item()

#unsupervised = True
evaluate_hyper_parameters()
pbounds={
  #  "log_learning_rate": (-8, -4),
    "log_train_freq": (0.0, 5),
  #  "ent_coef_ratio": (0, 0.00001),
   # "log_vf_coef": (-0.6931471805599453, 0.6931471805599453),
    #  "log_n_options": (2,5),
  #  "clip_range": (0.05,0.2),
 #   "log_n_epochs": (0,4),
}
optimizer = BayesianOptimization(
    f=evaluate_hyper_parameters,
    pbounds=pbounds,
    random_state=1,
)
#supervised_info = "_supervised_" if not unsupervised else "_unsupervised_"
#logger = JSONLogger(path="./eodppo_"+env_name+"_"+"_".join(pbounds.keys())+"_logs")
#optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
#optimizer.maximize( init_points=4,n_iter=32,)