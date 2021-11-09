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

import csv
import requests

use_trained = True


url = 'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY_EXTENDED&symbol=NIO&interval=1min&slice=year2month12&apikey=E1HH8I7PMXVTAGS9'
url = 'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY_EXTENDED&symbol=NIO&interval=1min&slice=year1month1&apikey=E1HH8I7PMXVTAGS9'
df = pd.read_csv(url)

#today = datetime.today().strftime('%Y-%m-%d')
#df = web.DataReader('NIO', data_source='av-intraday', start='2021-10-01', end=today, api_key='E1HH8I7PMXVTAGS9')
df.reset_index(inplace=True) # for some stupid reason we need this index column to be a number and not Date
df.sort_values('index', inplace=True)

# The algorithms require a vectorized environment to run
env = DummyVecEnv([lambda: StockTradingEnv(df)])

model = PPO('MlpPolicy', env, verbose=1)
if use_trained:
  model.load("nio-intraday-trained")
else:
  model.learn(total_timesteps=20000)
  model.save("nio-intraday-trained")

obs = env.reset()
for i in range(2000):
  action, _states = model.predict(obs)
  #action = [env.action_space.sample()]
  obs, rewards, done, info = env.step(action)
  print(rewards)
  env.render()
