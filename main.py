from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO

from envs.StockTradingEnv import StockTradingEnv

import pandas as pd


use_trained = True

function = 'TIME_SERIES_INTRADAY_EXTENDED'
symbol = 'NIO'
slice = 'year2month1' # oldest slice, for most recent use year1month12
from API_KEY import api_key
url = 'https://www.alphavantage.co/query?function=' + function + '&symbol=' + symbol + '&interval=1min&slice=' + slice + '&apikey=' + api_key

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
  #print('current net gain: {}'.format(rewards))
  env.render()
