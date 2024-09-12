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

from LSTM_config import load_config_list, save_config_list

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

import sys
sys.path.append('./')
# from Utils.metrics import mape, mae, mse, rmse, r_squered
from Utils.metrics import mape, mae, mse, rmse, r_squered
from RNN.Lset import TimeSeriesDataset
from RNN.lstm import StatelessLSTM , StatefulLSTM2 , StatefulLSTM1

# 하이퍼파라미터 정의
input_size, hidden_size, output_size, layers , period  = 1 , 2 , 32,  4 , 30

#15일, 30일, 60일, 90일
str = "15일"
selected_list = load_config_list(str)

results = []
# 모델 학습 및 예측
# for period in prediction_periods:
for input_size, hidden_size, output_size, layers, period in selected_list:
    # 데이터 스케일링 및 준비
    df = pd.read_csv('Data/2020-06-2021-05-31_all.csv')
    df.head()
    df['일시'] = df['Unnamed: 0']
    df = df.drop(columns='Unnamed: 0')
    df['일시'] = pd.to_datetime(df['일시'])
    df.set_index('일시',inplace=True)

    data = df
    prd_size = period
    tst_size = prd_size

    data_mw = data.copy()
    data_mw['rolling_avg'] = data['평균 수온(°C)'].rolling(12).mean()
    data_mw = data_mw.dropna()
    trn, tst = data_mw[:-period], data_mw[-period:]
    scaler = MinMaxScaler()
    scaler_ra = MinMaxScaler()
    trn_scaled, tst_scaled = trn.copy(), tst.copy()
    trn_scaled['평균 수온(°C)'] = scaler.fit_transform(trn['평균 수온(°C)'].to_numpy(np.float32).reshape(-1,1))
    trn_scaled['rolling_avg'] = scaler_ra.fit_transform(trn.rolling_avg.to_numpy(np.float32).reshape(-1,1))
    tst_scaled['평균 수온(°C)'] = scaler.transform(tst['평균 수온(°C)'].to_numpy(np.float32).reshape(-1,1))
    tst_scaled['rolling_avg'] = scaler_ra.transform(tst.rolling_avg.to_numpy(np.float32).reshape(-1,1))
    # print(tst_scaled.shape,trn_scaled.shape)
    trn_scaled = trn_scaled.to_numpy(np.float32)
    tst_scaled = tst_scaled.to_numpy(np.float32)


    # 한글 글꼴 설정
    plt.rcParams['font.family'] = 'NanumGothic'

    # 모델 정의
    batch_size = 128
    trn_x = torch.tensor(trn_scaled[:-1]).split(batch_size)
    trn_y = torch.tensor(trn_scaled[1:]).split(batch_size)
    tst_y = torch.tensor(tst_scaled)

    # 모델 학습
    rnn = StatefulLSTM2(input_size, hidden_size, output_size, layers)
    rnn.to(device)
    # 히든사이즈, 레이처갯수 증가시켜보기
    # 하이퍼 파라미터 변동 기록
    # 2 16 2 2
    # 2 32 2 3
    # 2 64 2 4
    
    # optim = torch.optim.Adam(rnn.parameters(), lr=0.0001)
    optim = torch.optim.AdamW(rnn.parameters(), lr=0.0001)
    # 옵티마이저 건드려보자.
    # Adam말고 다른거 lion 예전거들도 성능좋은거 있다.
    # 바꿔보자
    
    trn_predictions = []
    pbar = trange(1)
    for e in pbar:
        rnn.train()
        rnn.reset_state()
        trn_loss = .0
        for x, y in zip(trn_x, trn_y):
            x, y = x.to(device), y.to(device)
            optim.zero_grad()
            p = rnn(x)
            loss = F.mse_loss(p, y)
            loss.backward()
            optim.step()
            trn_loss += loss.item()
                    # 훈련 데이터에 대한 예측값 저장
            if e == len(pbar) - 1:  # 마지막 에폭에서만 저장
                trn_predictions.append(p.detach().cpu().numpy())
        trn_loss /= len(trn)-1
        
        rnn.eval()
        with torch.inference_mode():
            p = rnn.predict(y[-1:].to(device), len(tst_y))
            tst_loss = F.mse_loss(p, tst_y.to(device)).item()
        pbar.set_postfix({'trn_loss': trn_loss, 'tst_loss': tst_loss})

    # 성능 지표 계산
    prd = scaler.inverse_transform(p.cpu()[:, :1])
    mape_val = mape(prd, tst['평균 수온(°C)'].values.reshape(-1, 1))
    mae_val = mae(prd, tst['평균 수온(°C)'].values.reshape(-1, 1))
    mse_val = mse(prd, tst['평균 수온(°C)'].values.reshape(-1, 1))
    rmse_val = rmse(prd, tst['평균 수온(°C)'].values.reshape(-1, 1))
    r2_val = r_squered(prd, tst['평균 수온(°C)'].values.reshape(-1, 1))

    # 결과 저장
    tmp = (input_size, hidden_size, output_size, layers, period)
    results.append([tmp, mape_val, mae_val, mse_val, rmse_val, r2_val])

    trn_predictions = scaler.inverse_transform(np.concatenate(trn_predictions, axis=0)[:, :1])

    fig, axs = plt.subplots(1, 2, figsize=(15, 6))  # 1행 2열로 두 개의 그래프를 생성

    # 첫 번째 그래프 (학습 데이터)
    # axs[0].plot(np.concatenate(trn_predictions, axis=0)[:, 0].reshape(-1,1), label='Predicted')
    axs[0].plot(trn_predictions, label='Predicted')
    axs[0].plot(trn.iloc[1:, 0].values.reshape(-1, 1), label='Actual')
    axs[0].set_title(f'Train Data - P: {tmp}')
    axs[0].legend()

    # 두 번째 그래프 (테스트 데이터)
    axs[1].plot(prd, label='Predicted')
    axs[1].plot(tst['평균 수온(°C)'].to_numpy(), label='Actual')
    axs[1].set_title(f"Test Data\nLSTM(Stateful), MAPE:{mape_val:.4f}, MAE:{mae_val:.4f}, MSE:{mse_val:.4f}, RMSE:{rmse_val:.4f}, R2:{r2_val:.4f}")
    axs[1].legend()
    plt.tight_layout()  # 그래프 간 간격 조정
    plt.savefig('./LSTM-io_22'+ f'{tmp}.png')
    plt.show()

# 결과 DataFrame 생성
results_df = pd.DataFrame(results, columns=['Parmeter', 'MAPE', 'MAE', 'MSE', 'RMSE', 'R2'])
print(results_df)