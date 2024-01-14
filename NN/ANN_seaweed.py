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

import sys
sys.path.append('C:/Users/user/Desktop/TW_project') 
from Utils.metrics import mape,mae ,mse, rmse, r_squered
from NN.Nset import TimeSeriesDataset
from NN.ANN import Net


# years = '2016-01-2021-05-31'
years = '2020-06-2021-05-31'

df = pd.read_csv('Data/' + years + '_all.csv')
df['일시'] = df['Unnamed: 0']
df = df.drop(columns='Unnamed: 0')
df['일시'] = pd.to_datetime(df['일시'])
df.set_index('일시',inplace=True)
data = df
print(df.shape)
result = []     # metric 담을 lst



ord_lst = [[15, 5, 1, 400, 0.0001]]

for pred_size, lookback_size, forecast_size, epoch, lr in ord_lst:
	print(pred_size, lookback_size, forecast_size, epoch, lr)
	scaler = MinMaxScaler()
	trn_scaled = scaler.fit_transform(data[:-pred_size].to_numpy(dtype=np.float32)).flatten()
	tst_scaled = scaler.transform(data[-pred_size-lookback_size:].to_numpy(dtype=np.float32)).flatten()

	trn_df = data[:-pred_size].to_numpy(dtype=np.float32).flatten()
	tst_df = data[-pred_size - lookback_size:].to_numpy(dtype=np.float32).flatten()
	tst_y = tst_df[-pred_size:]

	### 스케일 안한 케이스
	trn_Ods = TimeSeriesDataset(trn_df, lookback_size, forecast_size)
	tst_Ods = TimeSeriesDataset(tst_df, lookback_size, forecast_size)

	trn_Odl = DataLoader(trn_Ods, batch_size=32, shuffle=True)
	tst_Odl = DataLoader(tst_Ods, batch_size=pred_size, shuffle=False)

	#### 스케일 한 케이스
	trn_ds = TimeSeriesDataset(trn_scaled, lookback_size, forecast_size)
	# tst_ds = TimeSeriesDataset(tst_scaled, lookback_size, forecast_size)

	trn_dl = DataLoader(trn_ds, batch_size=32, shuffle=True)
	# tst_dl = DataLoader(tst_ds, batch_size=pred_size, shuffle=False)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	net = Net(lookback_size, forecast_size, 512)
	net.to(device)
	optim = torch.optim.AdamW(net.parameters(), lr=lr)
	pbar = trange(epoch)
	losses = []
	trn_losses = []
	tst_rmse_losses = []

	for i in pbar:
		net.train()
		trn_loss = .0

		for x, y in trn_Odl: # 여기 2가지 케이스
			x, y = x.to(device), y.to(device)   # (32,15), (32, 5) in 32줄 (총 326개)
			p = net(x)    # (32, 5)인 predict output
			optim.zero_grad()
			loss = F.mse_loss(p, y)   # pred 와 target 차이
			loss.backward()
			optim.step()
			trn_loss += loss.item() * len(y)
		trn_loss = trn_loss / len(trn_Ods) # 여기 2가지 케이스

		net.eval()
		with torch.inference_mode():
			x, y = next(iter(tst_Odl)) # 여기 2가지 케이스
			x, y = x.to(device), y.to(device)     # x: input(15), y: target(5)
			p = net(x)                            # p: 예측 값(5)
			tst_loss = F.mse_loss(p, y) 
			y = y.cpu()
			p = p.cpu()    
			tst_mape = mape(p,y)
			tst_mae = mae(p,y)
			tst_rmse = rmse(p,y)
			tst_mse = mse(p,y)


		pbar.set_postfix({'trn_loss':trn_loss, 'tst_loss':tst_loss.item(), 'tst_mape':tst_mape.item(), 
												'tst_mae':tst_mae.item(), 'tst_rmse':tst_rmse.item()})
		tst_loss = tst_loss.cpu()  

		losses.append(tst_loss)
		trn_losses.append(trn_loss)
		tst_rmse_losses.append(tst_rmse)

	plot_start = 50     # 1부터 하면 훅 떨어지는거 그리느라 뒷부분이 잘 안보여서 확대한 것
	epochs_to_plot = range(plot_start, epoch)
	plt.figure(figsize=(10, 5))
	plt.title(f"Neural Network_({pred_size},{lookback_size},{forecast_size})_{epoch}_{lr}")
	plt.plot(epochs_to_plot, losses[plot_start:], label='tst_loss')
	plt.plot(epochs_to_plot, tst_rmse_losses[plot_start:], label='tst_rmse')
	plt.plot(epochs_to_plot, trn_losses[plot_start:], label='train_loss')
	plt.xticks(range(plot_start, epoch, 100))
	plt.legend()
	plt.savefig(f'./fig_ANN_0221_3/ANN_loss_{years}_({pred_size},{lookback_size},{forecast_size})_{epoch}_{lr}.png')
	plt.show


	preds = []
	x, y = trn_Ods[len(trn_Ods)-1]  #마지막의 input(15개), output(5개) 값을 가져옴
	# print(f'x: {x}, y: {y}\n')

	x = torch.tensor(x)
	# print(np.concatenate([x, y]).shape , x.shape, y.shape)

	net.eval()
	for _ in range(int(pred_size/forecast_size)):
		x = np.concatenate([x, y])[-lookback_size:]   # x = 20개[-15:] 즉, 15개

		with torch.inference_mode():    
			y = net(torch.tensor(x).cuda()).cpu()
			# print(f'y: {y}')
		preds.append(y)

	# print(f'\nlen(preds): {len(preds)}\n')  # len(preds) = 4 개 (for문 range만큼)

	preds = np.concatenate(preds)  # 예측 결과값을 하나의 Numpy 배열로 병합
	# print(f'preds: {preds}')    # 4개(5개씩)를 합치므로 4x5 = 20개

	MAPE = mape(preds,tst_y)
	MAE = mae(preds,tst_y)
	MSE = mse(preds,tst_y)
	RMSE = rmse(preds,tst_y)
	R2 = r_squared(preds,tst_y)
	result.append([(pred_size, lookback_size, forecast_size, epoch, lr), MAPE, MAE, MSE, RMSE, R2])

	print(p.shape)
	print(preds.shape)
	print(tst_y.shape)

	# y = y.numpy()
	# y = y.flatten()
	# preds, tst_y

	plt.figure(figsize=(10, 5))
	plt.title(f"NN_({pred_size},{lookback_size},{forecast_size})_{epoch}_MAPE:{MAPE:.3f}, MAE:{MAE:.3f}, MSE:{MSE:.3f}, RMSE:{RMSE:.3f}, R2:{R2:.3f}")
	plt.plot(range(pred_size), tst_y, label="True")
	plt.plot(range(pred_size), preds, label="Prediction")
	plt.legend()
	plt.savefig(f'./fig_ANN_0221_3/ANN_{years}_({pred_size},{lookback_size},{forecast_size})_{epoch}_{lr}_hidden1.png')
	plt.show()
	
result_df = pd.DataFrame(result, columns=['order', 'MAPE', 'MAE', 'MSE', 'RMSE', 'R2'])
result_df.set_index('order', inplace=True)
# result_df.to_csv(f'./result/ANN_result_{years}_hidden1.csv')
result_df.to_csv(f'./fig_ANN_0221_3/ANN_result_{years}_hidden1_3.csv')
