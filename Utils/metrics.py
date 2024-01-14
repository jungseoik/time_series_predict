
import numpy as np
from sklearn.metrics import r2_score

def mape(y_pred, y_true):
  return (np.abs(y_pred - y_true)/y_true).mean() * 100

def mae(y_pred, y_true):
  return np.abs(y_pred - y_true).mean()

def mse(y_pred, y_true):
  return np.mean((np.array(y_true) - np.array(y_pred)) ** 2)

def rmse(y_pred, y_true):
  return np.sqrt(mse(y_pred, y_true))


def r_squered(y_pred, y_true):
  return r2_score(y_true, y_pred)

def r2_score(y_pred, y_true):
  ss_tot = np.sum((y_true - np.mean(y_true))**2)
  ss_res = np.sum((y_true - y_pred)**2)
  r2 = 1 - (ss_res / ss_tot)
  return r2