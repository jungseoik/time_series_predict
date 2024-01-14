import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeSeriesDataset(torch.utils.data.Dataset):
  def __init__(self, ts:np.array, lookback_size:int, shift_size:int):
    self.lookback_size = lookback_size
    self.shift_size = shift_size
    self.data = ts

  def __len__(self):
    return len(self.data) - self.lookback_size - self.shift_size + 1

  def __getitem__(self, i):
    idx = (i+self.lookback_size)
    look_back = self.data[i:idx]
    forecast = self.data[i+self.shift_size:idx+self.shift_size]

    return look_back, forecast