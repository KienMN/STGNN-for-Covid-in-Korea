import pandas as pd
import os
from ..utils import listify

def load_df(filename):
  module_path = os.path.dirname(__file__)
  return pd.read_csv(module_path + '/data/' + filename)

def load_province_temporal_data(provinces='Seoul', status='New'):
  filename = '101_DT_COVID19_temporal_data_eng.csv'
  df = load_df(filename)
  df = df.drop(['Item', 'Unit'], axis=1)
  provinces = listify(provinces)
  temporal_data = {}
  for p in provinces:
    temporal_data[p] = df[(df['Category'] == p) & (df['Status'] == status)].iloc[0, 2:].tolist()
  # print(temporal_data)
  return pd.DataFrame(temporal_data, index=pd.to_datetime(df.columns[2:]), dtype='float32')