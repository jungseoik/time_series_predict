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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

class NetMulti(nn.Module):
  def __init__(self, d_in, d_out, d_hidden, c_in, activation=F.relu):
    super().__init__()
    self.lin1 = nn.Linear(d_in*c_in, d_hidden)
    self.lin2 = nn.Linear(d_hidden, d_out*c_in)
    self.activation = activation
    self.c_in = c_in
    self.d_out = d_out

  def forward(self, x):
    x = x.flatten(1)
    x = self.lin1(x)
    x = self.activation(x)
    x = self.lin2(x).reshape(-1, self.d_out, self.c_in)
    return x