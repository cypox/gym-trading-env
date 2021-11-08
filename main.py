import math
import gym
import json
import datetime as dt

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO

from envs.StockTradingEnv import StockTradingEnv

import pandas as pd
import pandas_datareader as web

from datetime import datetime

today = datetime.today().strftime('%Y-%m-%d')
df = web.DataReader('TSLA', data_source='av-daily', start='2010-01-01', end=today, api_key='E1HH8I7PMXVTAGS9')
df.reset_index(inplace=True) # for some stupid reason we need this index column to be a number and not Date
df.sort_values('index', inplace=True)

# The algorithms require a vectorized environment to run
env = DummyVecEnv([lambda: StockTradingEnv(df)])

model = PPO('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=2000)

obs = env.reset()
for i in range(2000):
  action, _states = model.predict(obs)
  #action = [env.action_space.sample()]
  obs, rewards, done, info = env.step(action)
  env.render()
