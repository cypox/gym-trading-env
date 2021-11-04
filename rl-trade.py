import math
import gym
import json
import pandas as pd
import pandas_datareader as web
from datetime import datetime
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.evaluation import evaluate_policy
import tensorflow as tf

from envs.StockTradingEnv import StockTradingBase


use_saved_model = True
use_saved_data = True

if use_saved_data:
    df = pd.read_pickle('TSLA.pkl')
else:
    # Load the dataset
    today = datetime.today().strftime('%Y-%m-%d')
    df = web.DataReader('TSLA', data_source='av-daily', start='2010-01-01', end=today, api_key='E1HH8I7PMXVTAGS9')
    df.reset_index(inplace=True) # for some stupid reason we need this index column to be a number and not Date
    df.sort_values('index', inplace=True)
    df.to_pickle('TSLA.pkl')

# The algorithms require a vectorized environment to run
env = DummyVecEnv([lambda: StockTradingBase(df)])

# Instantiate the agent
model = PPO('MlpPolicy', env, verbose=1)

if use_saved_model:
    # Load the trained agent
    model = PPO.load("ppo_trading")
else:
    # Train the agent
    model.learn(total_timesteps=2000)
    # Save the agent
    model.save("ppo_trading")
    #del model  # delete trained model to demonstrate loading

# Evaluate the agent
#mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)

# Enjoy trained agent
obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    #action = [env.action_space.sample()]
    obs, rewards, dones, info = env.step(action)
    env.render()
