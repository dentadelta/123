import gym
import delta_gym  #The magic is inside the delta_gym 
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import warnings
warnings.filterwarnings("ignore")
import optuna
import gym


def optimize_ppo(trial):
    return {
        'n_steps': trial.suggest_int('n_steps', 2048, 8192),
        'gamma': trial.suggest_uniform('gamma', 0.8, 0.99999),
        'learning_rate': trial.suggest_uniform('learning_rate', 1e-5, 1e-3),
        'clip_range': trial.suggest_uniform('clip_range', 0.1, 0.4),
        'gae_lambda': trial.suggest_uniform('gae_lambda', 0.8, 0.99),
        'max_grad_norm': trial.suggest_uniform('max_grad_norm', 0.1, 0.6),
        'vf_coef': trial.suggest_uniform('vf_coef', 0.1, 0.9),
        'ent_coef': trial.suggest_uniform('ent_coef', 0.0, 0.1),
        'batch_size': trial.suggest_int('batch_size', 64, 256),
    }

def optimize_agent(trial):
    model_parameter = optimize_ppo(trial)
    env = gym.make('deltaenv-v0',number_of_structures=2,number_of_components=2)
    env = make_vec_env(lambda: env, n_envs=1)


    model = PPO('MlpPolicy',env=env,tensorboard_log="./tensorboard_log/",verbose=0,**model_parameter)
    model.learn(total_timesteps=100000, tb_log_name=f'Optuna_PPO_{trial.number}')
    mean_reward,_ = evaluate_policy(model, env, n_eval_episodes=12)
    env.close()
        
    model.save("ppo_model{}".format(trial.number))
    return mean_reward


if __name__ == "__main__":
    study = optuna.create_study(direction='maximize',study_name='ppo_optimization')
    study.optimize(optimize_agent, n_trials=200, n_jobs=10)
