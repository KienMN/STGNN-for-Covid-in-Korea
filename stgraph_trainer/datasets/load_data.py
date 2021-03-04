import pandas as pd
import os
from ..utils import listify

def load_df(filename):
  """Load dataframe from a csv file with `filename` in `data` directory."""
  module_path = os.path.dirname(__file__)
  return pd.read_csv(module_path + '/data/' + filename)

def load_province_temporal_data(provinces='Seoul', status='New'):
  """Load temporal data for province(s) and status from original data file."""
  filename = '101_DT_COVID19_temporal_data_eng.csv'
  df = load_df(filename)
  df = df.drop(['Item', 'Unit'], axis=1)
  provinces = listify(provinces)
  temporal_data = {}
  for p in provinces:
    temporal_data[p] = df[(df['Category'] == p) & (df['Status'] == status)].iloc[0, 2:].tolist()
  return pd.DataFrame(temporal_data,
                      index=pd.to_datetime(df.columns[2:]),
                      dtype='float32')

def load_province_coordinates():
  """Load coordiantes of provinces."""
  filename = 'Region.csv'
  region_df = load_df(filename)
  province_coords = region_df[region_df.city == region_df.province][['province', 'latitude', 'longitude']].iloc[:-1]
  # Swap Incheon and Gwangju
  temp_row = province_coords.iloc[3].copy()
  province_coords.iloc[3] = province_coords.iloc[4].copy()
  province_coords.iloc[4] = temp_row
  return province_coords.reset_index(drop=True)