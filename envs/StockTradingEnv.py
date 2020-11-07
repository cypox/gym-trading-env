from .StockTradingBase import StockTradingBase
from .TradingGraph import TradingGraph

class StockTradingEnv(StockTradingBase):
  def __init__(self, df):
    super(StockTradingEnv, self).__init__(df)
    self.viewer = None

  def render(self, mode='human', **kwargs):
    pass
    if mode == 'human':
      if self.viewer == None:
        self.viewer = TradingGraph(self.df, kwargs.get('title', None))
      self.viewer.render(self.current_step,
                        self.net_worth)
