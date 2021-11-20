import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import mplfinance as mpf

from datetime import datetime

VOLUME_CHART_HEIGHT = 0.33

UP_COLOR = '#27A59A'
DOWN_COLOR = '#EF534F'
UP_TEXT_COLOR = '#73D3CC'
DOWN_TEXT_COLOR = '#DC2C27'

class TradingGraph:
  """A stock trading visualization using matplotlib made to render OpenAI gym environments"""

  def __init__(self, df, title=None):
    self.df = df
    self.net_worths = np.zeros(len(df['index']))

    # Create a figure on screen and set the title
    fig = plt.figure()
    fig.suptitle(title)

    self.net_worth_ax = fig.add_subplot(3,1,1)
    self.price_ax = fig.add_subplot(3,1,2,sharex=self.net_worth_ax)
    self.volume_ax = fig.add_subplot(3,1,3)

    # Add padding to make graph easier to view
    plt.subplots_adjust(wspace=0.2, hspace=0)

    # Show the graph without blocking the rest of the program
    plt.show(block=False)

  def _render_net_worth(self, current_step, net_worth, step_range, dates):
    # Clear the frame rendered last step
    self.net_worth_ax.clear()

    # Format the data
    net_worth_df = self.df.iloc[step_range].copy()
    net_worth_df.loc[:, 'close'] = self.net_worths[step_range]
    net_worth_df.loc[:, 'index'] = pd.to_datetime(self.df['index'])
    net_worth_df.set_index('index', inplace=True)

    # Plot net worths
    mpf.plot(net_worth_df, type='line', hlines=dict(hlines=[1000],colors=['r'],linestyle='-.'), ax=self.net_worth_ax, linecolor='#00ff00', ylabel='Net Worth')

    #self.net_worth_ax.xaxis.tick_top()

    # zoom y-axis to current window
    ymin = min(990, min(net_worth_df['close']) - 10)
    ymax = max(1010, max(net_worth_df['close']) + 10)
    self.net_worth_ax.set_ylim(ymin=ymin, ymax=ymax)

    #last_date = self.df['index'].values[current_step]
    last_date = self.net_worth_ax.get_xticks()[-2]
    last_net_worth = self.net_worths[current_step]

    # Annotate the current net worth on the net worth graph
    self.net_worth_ax.annotate('{0:.2f}'.format(net_worth), (last_date, last_net_worth),
                                xytext=(last_date, last_net_worth),
                                bbox=dict(boxstyle='round', fc='w', ec='k', lw=1),
                                color="black",
                                fontsize="small")

    # Add space above and below min/max net worth
    #self.net_worth_ax.set_ylim(min(self.net_worths[np.nonzero(self.net_worths)]) / 1.25, max(self.net_worths) * 1.25)

  def _render_price(self, current_step, net_worth, dates, step_range):
    self.price_ax.clear()
    self.volume_ax.clear()

    # Format the data
    price_range = self.df.iloc[step_range].copy()
    price_range.loc[:, 'index'] = pd.to_datetime(price_range['index'])
    price_range.set_index('index', inplace=True)

    #last_date = self.df['index'].values[current_step]
    last_date = self.net_worth_ax.get_xticks()[-2]
    last_close = self.df['close'].values[current_step]
    last_high = self.df['high'].values[current_step]

    # Print the current price to the price axis
    self.price_ax.annotate('{0:.2f}'.format(last_close), (last_date, last_close),
                            xytext=(last_date, last_high),
                            bbox=dict(boxstyle='round', fc='w', ec='k', lw=1),
                            color="black",
                            fontsize="small")

    # Plot price using mplfinance candlesticks after scaling axis otherwise, the scaling would be visible each frame
    mpf.plot(price_range, type='candle', style='charles', ax=self.price_ax, volume=self.volume_ax)

  def render(self, current_step, net_worth, window_size=40):
    self.net_worths[current_step] = net_worth

    window_start = max(current_step - window_size, 0)
    step_range = range(window_start, current_step + 1)

    # Format dates
    dates = np.array([x for x in self.df['index'].values[step_range]])

    self._render_net_worth(current_step, net_worth, step_range, dates)
    self._render_price(current_step, net_worth, dates, step_range)
    #self._render_trades(current_step, trades, step_range)

    # Hide duplicate net worth date labels
    plt.setp(self.net_worth_ax.get_xticklabels(), visible=False)

    # Necessary to view frames before they are unrendered
    plt.pause(0.0001)

  def close(self):
    plt.close()
