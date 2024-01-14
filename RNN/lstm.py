
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


class StatelessLSTM(nn.Module):
  def __init__(self, input_size, hidden_size, output_size, num_layers):
    super().__init__()
    self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
    self.head = nn.Linear(hidden_size, output_size)

  def forward(self, x):
    x, _ = self.rnn(x)   # state will be set to be zeros by default
    # x.shape == (batch_size, sequence_length, hidden_size)
    x = self.head(x)  # (batch_size, sequence_length, output_size)
    return F.sigmoid(x)

  def predict(self, x, steps, state=None):
    output = []
    for i in range(steps):
      x = self.forward(x)
      output.append(x[-1:])
    return torch.concat(output, 0)
  

class StatefulLSTM2(nn.Module):
  def __init__(self, input_size, hidden_size, output_size, num_layers):
    super().__init__()
    self.reset_state()
    self.rnn = nn.LSTM(input_size, hidden_size, num_layers)
    self.head = nn.Linear(hidden_size, output_size)

  def reset_state(self, state=None):
    self.state = state

  def forward(self, x):
    assert x.dim() == 2   # (sequence_length, input_size)
    
    if self.state is None:
      x, (hn, cn) = self.rnn(x)   # state will be set to be zeros by default
    else:
      x, (hn, cn) = self.rnn(x, self.state)   # pass the saved state
      
    # Check the shape of x here
    # print("Shape of x after LSTM:", x.shape)  
    # x.shape == (sequence_length, hidden_size)
    self.reset_state((hn.detach(), cn.detach()))  # save the state
    x = self.head(x)  # (sequence_length, output_size)
    return F.sigmoid(x)

  def predict(self, x0, steps, state=None):
    if state is not None:
      self.reset_state(state)
      
    output = []
    x = x0.reshape(1,-1)
    for i in range(steps):
      x = self.forward(x)
      output.append(x)
    return torch.concat(output, 0)
  
class StatefulLSTM1(nn.Module):
  def __init__(self, input_size, hidden_size, output_size, num_layers):
    super().__init__()
    self.reset_state()
    self.rnn = nn.LSTM(input_size, hidden_size, num_layers)
    self.head = nn.Linear(hidden_size, output_size)

  def reset_state(self, state=None):
    self.state = state

  def forward(self, x):
    assert x.dim() == 2   # (sequence_length, input_size)
    
    if self.state is None:
      x, (hn, cn) = self.rnn(x)   # state will be set to be zeros by default
    else:
      x, (hn, cn) = self.rnn(x, self.state)   # pass the saved state
      
    # Check the shape of x here
    # print("Shape of x after LSTM:", x.shape)  
    # x.shape == (sequence_length, hidden_size)
    self.reset_state((hn.detach(), cn.detach()))  # save the state
    x = self.head(x)  # (sequence_length, output_size)
    return F.sigmoid(x)

  def predict(self, x0, steps, state=None):
    if state is not None:
      self.reset_state(state)
      
    output = []
    x = x0.reshape(1,-1)
    for i in range(steps):
      x = self.forward(x)
      output.append(x)
    return torch.concat(output, 0)