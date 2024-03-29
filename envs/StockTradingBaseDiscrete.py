import random
import json
import gym
from gym import spaces
import numpy as np

MAX_ACCOUNT_BALANCE = 2147483647
MAX_NUM_SHARES = 2147483647
MAX_SHARE_PRICE = 100
MAX_OPEN_POSITIONS = 5
MAX_STEPS = 20000

INITIAL_ACCOUNT_BALANCE = 1000


class StockTradingBaseDiscrete(gym.Env):
  """A stock trading environment for OpenAI gym"""
  metadata = {'render.modes': ['human']}

  def __init__(self, df):
    super(StockTradingBaseDiscrete, self).__init__()

    self.df = df
    self.reward_range = (0, MAX_ACCOUNT_BALANCE)

    # Actions of the format Buy, Sell, Hold
    self.action_space = spaces.Discrete(3)

    # Prices contains the OHCL values for the last five prices
    self.observation_space = spaces.Box(low=0, high=1, shape=(5, 6), dtype=np.float16)

  def _next_observation(self):
    # Get the stock data points for the last 5 days and scale to between 0-1
    frame = np.array([
      self.df.loc[self.current_step: self.current_step +
                  5, 'open'].values / MAX_SHARE_PRICE,
      self.df.loc[self.current_step: self.current_step +
                  5, 'high'].values / MAX_SHARE_PRICE,
      self.df.loc[self.current_step: self.current_step +
                  5, 'low'].values / MAX_SHARE_PRICE,
      self.df.loc[self.current_step: self.current_step +
                  5, 'close'].values / MAX_SHARE_PRICE,
      self.df.loc[self.current_step: self.current_step +
                  5, 'volume'].values / MAX_NUM_SHARES,
    ])

    # Append additional data and scale each value to between 0-1
    obs = np.append(frame, [[
      self.balance / MAX_ACCOUNT_BALANCE,
      self.max_net_worth / MAX_ACCOUNT_BALANCE,
      self.shares_held / MAX_NUM_SHARES,
      self.cost_basis / MAX_SHARE_PRICE,
      self.total_shares_sold / MAX_NUM_SHARES,
      self.total_sales_value / (MAX_NUM_SHARES * MAX_SHARE_PRICE),
    ]], axis=0)
    # remove additional data
    obs = np.array(frame)

    return obs

  def _take_action(self, action):
    # Set the current price to a random price within the time step
    current_price = random.uniform(self.df.loc[self.current_step, "open"], self.df.loc[self.current_step, "close"])

    action_type = action
    amount = 1

    if action_type == 0: # hold
      #print('holding @ {}'.format(current_price))
      pass
    elif action_type == 1: # buy 1 share if possible
      #print('buying @ {}'.format(current_price))
      total_possible = int(self.balance / current_price)
      shares_bought = amount if total_possible >= amount else 0
      prev_cost = self.cost_basis * self.shares_held
      additional_cost = amount * current_price

      if shares_bought > 0:
        self.balance -= additional_cost
        self.cost_basis = (prev_cost + additional_cost) / (self.shares_held + shares_bought)
        self.shares_held += shares_bought

    elif action_type == 2: # sell 1 share if possible
      #print('selling @ {}'.format(current_price))
      shares_sold = amount if self.shares_held >= amount else self.shares_held
      self.balance += shares_sold * current_price
      self.shares_held -= shares_sold
      self.total_shares_sold += shares_sold
      self.total_sales_value += shares_sold * current_price

    self.net_worth = self.balance + self.shares_held * current_price

    if self.net_worth > self.max_net_worth:
      self.max_net_worth = self.net_worth

    if self.shares_held == 0:
      self.cost_basis = 0

  def step(self, action):
    # Execute one time step within the environment
    self._take_action(action)

    self.current_step += 1

    if self.current_step > len(self.df.loc[:, 'open'].values) - 6:
      self.current_step = 0

    delay_modifier = (self.current_step / MAX_STEPS)
    delay_modifier = 1

    reward = self.net_worth * delay_modifier - INITIAL_ACCOUNT_BALANCE
    done = self.net_worth <= 0

    obs = self._next_observation()

    return obs, reward, done, {}

  def reset(self):
    # Reset the state of the environment to an initial state
    self.balance = INITIAL_ACCOUNT_BALANCE
    self.net_worth = INITIAL_ACCOUNT_BALANCE
    self.max_net_worth = INITIAL_ACCOUNT_BALANCE
    self.shares_held = 0
    self.cost_basis = 0
    self.total_shares_sold = 0
    self.total_sales_value = 0

    # Set the current step to a random point within the data frame
    self.current_step = random.randint(0, len(self.df.loc[:, 'open'].values) - 6)

    return self._next_observation()

  def render(self, mode='human', close=False):
    # Render the environment to the screen
    profit = self.net_worth - INITIAL_ACCOUNT_BALANCE

    print(f'Step: {self.current_step}')
    print(f'Balance: {self.balance}')
    print(f'Shares held: {self.shares_held} (Total sold: {self.total_shares_sold})')
    print(f'Avg cost for held shares: {self.cost_basis} (Total sales value: {self.total_sales_value})')
    print(f'Net worth: {self.net_worth} (Max net worth: {self.max_net_worth})')
    print(f'Profit: {profit}')
