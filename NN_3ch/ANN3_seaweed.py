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
from Utils.metrics import mape,mae ,mse, rmse, r_squered, r2_score
from NN_3ch.Nset3 import TimeSeriesDatasetMulti, generate_time_series_loaders
from NN_3ch.ANN3 import  NetMulti

order_lst = [[30, 90, 15, 512, 3, 500, 0.0001], [30, 90, 15, 512, 3, 1500, 0.0001]]
result = []

for pred_size, lookback_size, forecast_size, hidden_dim, channel_size, epoch, lr in order_lst:
	print(f'pred_size: {pred_size}, lookback_size: {lookback_size}, forecast_size: {forecast_size}, hidden_dim: {hidden_dim}, channel_size: {channel_size}, epoch: {epoch}, lr: {lr}')

	tst_y, trn_Ods, tst_Ods, trn_Odl, tst_Odl, trn_ds, trn_dl = generate_time_series_loaders(pred_size, lookback_size, forecast_size)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	
	net = NetMulti(lookback_size, forecast_size, hidden_dim, channel_size)
	net.to(device)

	optim = torch.optim.AdamW(net.parameters(), lr=lr)
	pbar = trange(epoch)
	
	losses = []
	trn_losses = []
	tst_rmse_losses = []

	for i in pbar:
		net.train()
		trn_loss = .0

		for x, y in trn_Odl:
			x, y = x.to(device), y.to(device)
			p = net(x)
			optim.zero_grad()
			loss = F.mse_loss(p, y)
			loss.backward()
			optim.step()
			trn_loss += loss.item() * len(y)
		trn_loss = trn_loss / len(trn_Ods)

		net.eval()
		with torch.inference_mode():
			x, y = next(iter(tst_Odl))
			x, y = x.to(device), y.to(device)
			p = net(x)
			tst_loss = F.mse_loss(p, y) 
			y = y.cpu()
			p = p.cpu()    
			tst_mape = mape(p, y)
			tst_mae = mae(p, y)
			tst_rmse = rmse(p, y)
			tst_mse = mse(p, y)

		pbar.set_postfix({
			'Multi-channel trn_loss':trn_loss, 'tst_loss':tst_loss.item(), 'tst_mape':tst_mape.item(), 
			'tst_mae':tst_mae.item(), 'tst_rmse':tst_rmse.item()
		})
		tst_loss = tst_loss.cpu()  
		losses.append(tst_loss)
		trn_losses.append(trn_loss)
		tst_rmse_losses.append(tst_rmse)

	path = f'./NN_{channel_size}ch/model/model_ANN_{channel_size}multi.pth'
	torch.save(net.state_dict(), path)

	plot_start = 50
	epochs_to_plot = range(plot_start, epoch)
	plt.figure(figsize=(10, 5))
	plt.title(f"Neural Network {channel_size}Multi-channel_({pred_size},{lookback_size},{forecast_size})_{epoch}_{lr}")
	plt.plot(epochs_to_plot, losses[plot_start:], label='tst_loss')
	plt.plot(epochs_to_plot, tst_rmse_losses[plot_start:], label='tst_rmse')
	plt.plot(epochs_to_plot, trn_losses[plot_start:], label='train_loss')
	plt.xticks(range(plot_start, epoch, 100))
	plt.legend()
	plt.savefig(f'./NN_{channel_size}ch/fig/ANN_{channel_size}multi_loss_({pred_size},{lookback_size},{forecast_size})_{epoch}_{lr}.png')
	plt.show

	preds = []
	x, y = trn_Ods[len(trn_Ods)-1]

	net = NetMulti(lookback_size, forecast_size, hidden_dim, channel_size)
	net.load_state_dict(torch.load(path))

	net.eval()
	for _ in range(int(pred_size/forecast_size)):
		if y.shape == (1, forecast_size, channel_size):
			y = y.squeeze(0)
		x = np.concatenate([x, y])[-lookback_size:]
		x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)

		with torch.inference_mode():
			y = net(x_tensor).cpu()
		preds.append(y)

	preds = np.concatenate(preds)
	final_preds = preds[:,:,0].flatten()
	tst_y = tst_y[:,0]

	MAPE = mape(final_preds, tst_y)
	MAE = mae(final_preds, tst_y)
	MSE = mse(final_preds, tst_y)
	RMSE = rmse(final_preds, tst_y)
	R2 = r_squered(final_preds, tst_y)
	R2S = r2_score(final_preds, tst_y)
	result.append([(pred_size, lookback_size, forecast_size, epoch, lr), MAPE, MAE, MSE, RMSE, R2, R2S])

	plt.figure(figsize=(10, 5))
	plt.title(f"NN {channel_size}Multi-channel_({pred_size},{lookback_size},{forecast_size})_{epoch}_MAPE:{MAPE:.3f}, MAE:{MAE:.3f}, MSE:{MSE:.3f}, RMSE:{RMSE:.3f}, R2:{R2:.3f}, R2S:{R2S:.3f}")
	plt.plot(range(pred_size), tst_y, label="True")
	plt.plot(range(pred_size), final_preds, label="Prediction")
	plt.legend()
	plt.savefig(f'./NN_{channel_size}ch/result/ANN_{channel_size}multi_({pred_size},{lookback_size},{forecast_size})_{epoch}_{lr}.png')
	plt.show()

pd.set_option('float_format', '{:.4f}'.format)
result_df = pd.DataFrame(result, columns=['order', 'MAPE', 'MAE', 'MSE', 'RMSE', 'R2', 'R2S'])
result_df.set_index('order', inplace=True)
result_df.to_csv(f'./NN_{channel_size}ch/ANN_' + str(channel_size) + 'multi_result_' + '.csv')
result_df
