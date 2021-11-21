import pandas as pd


def download_data():
  function = 'TIME_SERIES_INTRADAY_EXTENDED'
  symbol = 'NIO'
  interval = '1min'
  slices = ['year2month1', 'year2month2', 'year2month3'] # oldest slice, for most recent use year1month12
  from .API_KEY import api_key
  df = None
  for slice in slices:
    url = 'https://www.alphavantage.co/query?function=' + function + '&symbol=' + symbol + '&interval=' + interval + '&slice=' + slice + '&apikey=' + api_key

    if df is None:
      df = pd.read_csv(url)
    else:
      df = df.append(pd.read_csv(url))

  df.reset_index(inplace=True)
  df.sort_values('index', inplace=True)

  df.to_csv('./data.csv')
  return df

def load_data():
  df = pd.read_csv('./data.csv')
  return df
