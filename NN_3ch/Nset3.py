import numpy as np
import pandas as pd
from tqdm.auto import trange
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-darkgrid')
from sklearn.preprocessing import MinMaxScaler
from tqdm.auto import trange

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import r2_score


class TimeSeriesDatasetMulti(Dataset):
  def __init__(self, ts:np.array, lookback_size:int, forecast_size:int):
    self.lookback_size = lookback_size
    self.forecast_size = forecast_size
    self.data = ts

  def __len__(self):
    return len(self.data) - self.lookback_size - self.forecast_size + 1

  def __getitem__(self, i):
    idx = (i + self.lookback_size)
    look_back = self.data[i:idx]
    forecast = self.data[idx:idx + self.forecast_size]
    return look_back, forecast
  
def rename_column(data):
  return data.rename(columns={'평균 수온(°C)':'temperature'}, inplace=True)
  
def generate_time_series_loaders(pred_size, lookback_size, forecast_size):

  years = '2020-06-2021-05-31'
  df = pd.read_csv('./data/' + years + '_all.csv')
  df['일시'] = df['Unnamed: 0']
  df = df.drop(columns='Unnamed: 0')
  df['일시'] = pd.to_datetime(df['일시'])
  df.set_index('일시',inplace=True)
  rename_column(df)
  data = df
  window_size = 12
  data['rolling_mean'] = data.rolling(window_size).mean()
  rename_column(data)
  data = data.dropna()
  data['temperature_diff'] = data['temperature'] - data['rolling_mean']
      
  trn_df = data[:-pred_size].to_numpy(dtype=np.float32)
  tst_df = data[-pred_size - lookback_size:].to_numpy(dtype=np.float32)
  tst_y = tst_df[-pred_size:]

  # 스케일 안 한 케이스
  trn_Ods = TimeSeriesDatasetMulti(trn_df, lookback_size, forecast_size)
  tst_Ods = TimeSeriesDatasetMulti(tst_df, lookback_size, forecast_size)

  trn_Odl = DataLoader(trn_Ods, batch_size=32, shuffle=True)
  tst_Odl = DataLoader(tst_Ods, batch_size=pred_size, shuffle=False)

  # 스케일 한 케이스
  scaler = MinMaxScaler()
  trn_scaled = scaler.fit_transform(data[:-pred_size].to_numpy(dtype=np.float32))
  tst_scaled = scaler.transform(data[-pred_size-lookback_size:].to_numpy(dtype=np.float32))

  trn_ds = TimeSeriesDatasetMulti(trn_scaled, lookback_size, forecast_size)
  # tst_ds = TimeSeriesDataset(tst_scaled, lookback_size, forecast_size)

  trn_dl = DataLoader(trn_ds, batch_size=32, shuffle=True)
  # tst_dl = DataLoader(tst_ds, batch_size=pred_size, shuffle=False)

  return tst_y, trn_Ods, tst_Ods, trn_Odl, tst_Odl, trn_ds, trn_dl