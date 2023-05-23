import torch
from torch import nn
from einops import rearrange
from einops.layers.torch import Rearrange

class RegLSTM(nn.Module):
    def __init__(self, input_dim, input_size, hidden_size, hidden_num_layers):
        super(RegLSTM, self).__init__()
        # 嵌入
        self.to_patch_embedding = self.to_patch_embedding = nn.Sequential(
            Rearrange('b 1 (n d) -> b 1 n d', n=input_size),
            nn.Linear(input_dim // input_size, input_dim // input_size)
        )
        # 定义LSTM
        self.rnn = nn.LSTM(input_dim // input_size, hidden_size, hidden_num_layers, batch_first=True)
        # 定义回归层网络，输入的特征维度等于LSTM的输出，输出维度为1
        self.reg = nn.Sequential(
            nn.Linear(hidden_size * input_size, 10)
        )

    def forward(self, x):
        x = rearrange(x, 'b l -> b 1 l')
        x = self.to_patch_embedding(x)
        x = x.squeeze(1)
        x, (ht, ct) = self.rnn(x)
        x = rearrange(x, 'b s l -> b (s l)')
        x = self.reg(x)
        return x
