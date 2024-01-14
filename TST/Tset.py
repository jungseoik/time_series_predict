import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchTSDataset(torch.utils.data.Dataset):
    def __init__(self, ts:np.array, patch_length:int=16, n_patches:int=6, prediction_length:int=4):
        self.P = patch_length # 패치길이가 엥간하면 16으로 (논문에서 항상16으로 했다함)
        self.N = n_patches # look back 사이즈에 따라서 달라지는데 64했을때 가장 좋았다.(논문에서)
        self.L = int(patch_length * n_patches / 2)  
        self.T = prediction_length #(논문에서도 여러가지로 해봤고, 여러가지로 해봐라, 780, 등등) # 문제에서 타겟으로 하는 길이 사용
        self.data = ts

    def __len__(self):
        return len(self.data) - self.L - self.T + 1

    def __getitem__(self, i):
        look_back = self.data[i:(i+self.L)]
        look_back = np.concatenate([look_back, look_back[-1]*np.ones(int(self.P / 2), dtype=np.float32)])
        x = np.array([look_back[i*int(self.P/2):(i+2)*int(self.P/2)] for i in range(self.N)])
        y = self.data[(i+self.L):(i+self.L+self.T)]
        return x, y
  

class PatchTSDatasetX(torch.utils.data.Dataset):
    def __init__(self, ts:np.array, patch_length:int=16, n_patches:int=6, prediction_length:int=4):
        self.P = patch_length # 패치길이가 엥간하면 16으로 (논문에서 항상16으로 했다함)
        self.N = n_patches # look back 사이즈에 따라서 달라지는데 64했을때 가장 좋았다.(논문에서)
        self.L = int(patch_length * n_patches / 2)  
        self.T = prediction_length #(논문에서도 여러가지로 해봤고, 여러가지로 해봐라, 780, 등등) # 문제에서 타겟으로 하는 길이 사용
        self.data = ts
    def __len__(self):
        return len(self.data) - self.L  + 1
    def __getitem__(self, i):
        look_back = self.data[i:(i+self.L)]
        look_back = np.concatenate([look_back, look_back[-1]*np.ones(int(self.P / 2), dtype=np.float32)])
        x = np.array([look_back[i*int(self.P/2):(i+2)*int(self.P/2)] for i in range(self.N)])
        return x