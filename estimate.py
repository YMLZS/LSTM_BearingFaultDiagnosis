import os
import torch
from model import RegLSTM
from torchsummary import summary


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    N = 4 #编码器个数
    input_dim = 1024
    seq_len = 16 #句子长度
    hidden_size = 128
    lr = 6E-5 #学习率

    net = RegLSTM(input_dim=input_dim, input_size=seq_len, hidden_size=hidden_size, hidden_num_layers=N)
    device = torch.device("cuda:0")
    net = net.to(device)

    input_data = torch.randn(1, 1024)
    input_data = input_data.squeeze() # 适应summary()的输入维度,因为summary会给输入拼接一个维度
    summary(net, input_data.size())
