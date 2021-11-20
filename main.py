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

use_trained = False


url = 'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY_EXTENDED&symbol=NIO&interval=1min&slice=year2month12&apikey=E1HH8I7PMXVTAGS9'
#url = 'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY_EXTENDED&symbol=NIO&interval=1min&slice=year1month1&apikey=E1HH8I7PMXVTAGS9'
df = pd.read_csv(url)

df.reset_index(inplace=True)
df.sort_values('index', inplace=True)

# The algorithms require a vectorized environment to run
env = DummyVecEnv([lambda: StockTradingEnv(df)])

model = PPO('MlpPolicy', env, verbose=1)
if use_trained:
  model.load("nio-intraday-trained", env=env)
else:
  model.learn(total_timesteps=20000)
  model.save("nio-intraday-trained")

obs = env.reset()
for i in range(2000):
  action, _states = model.predict(obs)
  #action = [env.action_space.sample()]
  obs, rewards, done, info = env.step(action)
  #print(rewards)
  env.render()
