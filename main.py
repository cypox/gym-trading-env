from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

from envs.StockTradingEnv import StockTradingEnv

import numpy as np
import pandas as pd


use_trained = False

function = 'TIME_SERIES_INTRADAY_EXTENDED'
symbol = 'NIO'
interval = '1min'
slice = 'year2month1' # oldest slice, for most recent use year1month12
from API_KEY import api_key
url = 'https://www.alphavantage.co/query?function=' + function + '&symbol=' + symbol + '&interval=' + interval + '&slice=' + slice + '&apikey=' + api_key

df = pd.read_csv(url)

df.reset_index(inplace=True)
df.sort_values('index', inplace=True)

# The algorithms require a vectorized environment to run
env = DummyVecEnv([lambda: StockTradingEnv(df)])

model = PPO('MlpPolicy', env, verbose=1, tensorboard_log="./nio_trading_tensorboard/")
if use_trained:
  model.load("nio-intraday-trained", env=env)
else:
  model.learn(total_timesteps=100000)
  model.save("nio-intraday-trained")

#mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

def dummy_strategy(obs):
  obs = obs[0]
  moving_average = np.mean(obs[1][1:])
  if obs[1][0] > moving_average:
    # sell high
    act = 2
  elif obs[1][0] < moving_average:
    # buy low
    act = 1
  else:
    # hodl
    act = 0
  return [act]

obs = env.reset()
for i in range(2000):
  action, _states = model.predict(obs)
  #action = dummy_strategy(obs)
  #action = [env.action_space.sample()]
  obs, rewards, done, info = env.step(action)
  print('current net gain: {}'.format(rewards))
  env.render()
