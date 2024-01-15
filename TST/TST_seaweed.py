import numpy as np
import pandas as pd
from tqdm.auto import trange
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-darkgrid')
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

from lion_pytorch import Lion
import sys
sys.path.append('C:/Users/user/Desktop/TW_project') 
from Utils.metrics import mape,mae ,mse, rmse, r_squered
from TST.Tset import PatchTSDataset, PatchTSDatasetX
from TST.tf import PatchTST
from TST.config import config_list

# 하이퍼파라미터 정의
# 트랜스포머 레이어 수(L)는 {3, 4, 5}로 변화시키고 모델 차원(D)은 {128, 256}으로 선택했습니다. 피드 포워드 네트워크의 내부 레이어는 F=2D입니다. 총 6가지 다른 모델 하이퍼파라미터 세트를 검토합니다
# d_model 값을 num_heads로 나누어 떨어지게 조정

patch_length, n_patches, prediction_length, tst_size, model_dim, heads, layer, epoch  = 16, 64, 90 ,90 , 512, 8 , 4 ,1000

list = [[4, 64, 30, 30, 512, 8, 4, 1]]
results = []

list = config_list("15일")


for patch_length, n_patches, prediction_length, tst_size, model_dim, num_heads, num_layers ,epoch in list:
    # 데이터 스케일링 및 준비

    df = pd.read_csv('Data/2020-06-2021-05-31_all.csv')
    df.head()
    df['일시'] = df['Unnamed: 0']
    df = df.drop(columns='Unnamed: 0')
    df['일시'] = pd.to_datetime(df['일시'])
    df.set_index('일시',inplace=True)
    data = df



    window_size = int(patch_length * n_patches / 2)

    scaler = MinMaxScaler()
    trn_scaled = scaler.fit_transform(data[:-tst_size].to_numpy(dtype=np.float32)).flatten()
    tst_scaled = scaler.transform(data[-tst_size-window_size:].to_numpy(dtype=np.float32)).flatten()
    # 길이가 너무짧은데 패치 크기나, 예측 사이즈가 너무크면 길이가 맞지 않아서 오류뜸
    # 파라미터를 여유롭게 조정할 것.
    trn_ds = PatchTSDataset(trn_scaled, patch_length, n_patches ,prediction_length)
    tst_ds = PatchTSDataset(tst_scaled, patch_length, n_patches, prediction_length)
  
    trn_dl = torch.utils.data.DataLoader(trn_ds, batch_size=32, shuffle=True)
    tst_dl = torch.utils.data.DataLoader(tst_ds, batch_size=tst_size, shuffle=False)
        
    model = PatchTST(n_patches, patch_length, model_dim, num_heads, num_layers, prediction_length)
    model.to(device)
    # optim = torch.optim.Adam(model.parameters(), lr=0.0001)
    # optim = Lion(model.parameters(), lr=0.0001)
    optim = torch.optim.AdamW(model.parameters(), lr=0.0001)
    trn_losses = []
    tst_losses = []
    pbar = trange(epoch)
    for _ in pbar:
        model.train()
        trn_loss = 0.
        for x,y in trn_dl:
            x, y = x.to(device), y.to(device)
            p = model(x)
            optim.zero_grad()
            loss = F.mse_loss(p, y)
            loss.backward()
            optim.step()
            trn_loss += loss.item()*len(x)
        trn_loss = trn_loss / len(trn_ds)

        model.eval()
        with torch.inference_mode():
            x, y = next(iter(tst_dl))
            x, y = x.to(device), y.to(device)
            p = model(x)
            tst_loss = F.mse_loss(p,y)
        trn_losses.append(trn_loss)
        tst_losses.append(tst_loss.item())
        pbar.set_postfix({'loss':trn_loss, 'tst_loss':tst_loss.item()})


    trn_ds = PatchTSDatasetX(trn_scaled[-window_size:], patch_length, n_patches ,prediction_length)
    trn_dl = torch.utils.data.DataLoader(trn_ds, batch_size=1, shuffle=True)
    model.eval()
    x = next(iter(trn_dl))
    
    with torch.no_grad():
        prediction = model(x.to(device))
    p = scaler.inverse_transform(prediction.cpu())
    with torch.inference_mode():
        _, y = next(iter(tst_dl))
        y =  y.to(device)
    y = scaler.inverse_transform(y.cpu())
    y = y.reshape(1,-1).flatten()
    p = p.flatten()

    mape_ = mape(p,y)
    mae_ = mae(p,y)
    mse_ = mse(p,y)
    rmse_ = rmse(p,y)
    r2 = r_squered(p,y)

    tmp = (patch_length, n_patches, prediction_length, tst_size ,model_dim, num_heads, num_layers  ,epoch )
    results.append([tmp, mape_, mae_, mse_, rmse_, r2])

    fig, axs = plt.subplots(1, 2, figsize=(15, 6))  # 1행 2열로 두 개의 그래프를 생성
    # plt.figure(figsize=(10, 5))
    axs[0].plot(trn_losses, label='Train Loss')
    axs[0].plot(tst_losses, label='Test Loss')
    axs[0].legend()

    axs[1].set_title(f"PatchTST 2021-2022, MAPE:{mape_:.4f}, MAE:{mae_:.4f}, MSE:{mse_:.4f}, RMSE:{rmse_:.4f}, R2:{r2}")
    axs[1].plot(range(prediction_length), y, label="True")
    axs[1].plot(range(prediction_length), p, label="Prediction")
    axs[1].legend()
    plt.savefig('./Patch_TST'+ f'{tmp}.png')
    plt.tight_layout()  # 그래프 간 간격 조정
    plt.show()
    # 결과 DataFrame 생성

results_df = pd.DataFrame(results, columns=['Parmeter', 'MAPE', 'MAE', 'MSE', 'RMSE', 'R2'])
print(results_df)
results_df.to_csv('Patch_TST_2021-05-2022-05.csv')