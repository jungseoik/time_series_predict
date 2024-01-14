import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchTST(nn.Module):
  def __init__(self, n_token, input_dim, model_dim, num_heads, num_layers, output_dim):
    super(PatchTST, self).__init__()
    self.patch_embedding = nn.Linear(input_dim, model_dim)    
    # Input Embedding # 인풋 디멘션 토큰1개의차원 , 모델디멘션이 임베딩 벡터
    # 모델 dim을 128로 주면 좋다라는 이야기도 존재한다.
    # 실험결과가 해당 논문에 존재함 참고

    self._pos = torch.nn.Parameter(torch.randn(1,1,model_dim))  # Positional Embedding

    encoder_layers = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, batch_first=True)
    self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
    # num layer 인코더 몇개붙일지의 갯수
    
    self.output_layer = nn.Linear(model_dim * n_token, output_dim)
    # 4일 예측하면 아웃풋 디멘션,
  def forward(self, x):
    # x shape: (batch_size, n_token, token_size)
    x = self.patch_embedding(x)   # (batch_size, n_token, model_dim)
    x = x + self._pos
    x = self.transformer_encoder(x)   # (batch_size, n_token, model_dim)
    x = x.view(x.size(0), -1)       # (batch_size, n_token * model_dim)
    output = self.output_layer(x)   # (batch_size, out_dim =4 patch_size == 4)
    return F.sigmoid(output)