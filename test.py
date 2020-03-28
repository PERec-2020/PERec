import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

from torch.autograd import Variable
# torch.manual_seed(1)

# # 输入输出维度
# lstm=nn.LSTM(3,3)

# inputs=[autograd.Variable(torch.randn((1,3))) for _ in range(5)]

# hidden=(autograd.Variable(torch.randn(1,1,3)),autograd.Variable(torch.randn((1,1,3))))

# for i in inputs:
#     out,hidden = lstm(i.view(1,1,-1),hidden)
#     print("out",out)
#     print("hidden",hidden)


# 输入维度 50，隐层100维，两层
lstm_seq = nn.LSTM(2, 4, num_layers=1, bidirectional=False)
# 查看网络的权重，ih和hh，共2层，所以有四个要学习的参数
# lstm_seq.weight_hh_l0.size(), 
# lstm_seq.weight_hh_l1.size(), 
# lstm_seq.weight_ih_l0.size(), 
# lstm_seq.weight_ih_l1.size()
# q1： 输出的size是多少？ 都是torch.Size([400, 100]

# 输入序列seq= 10，batch =3，输入维度=50
lstm_input = Variable(torch.randn(10, 3, 2))

permute_lstm_input=lstm_input.permute(1,0,2)
print("permute",permute_lstm_input.size())
print("lstm_input",lstm_input.size())

out, (h, c) = lstm_seq(lstm_input)  # 使用默认的全 0 隐藏状态
# q1：out和(h,c)的size各是多少？out：(10*3*100)，（h,c）：都是(2*3*100)
# q2:out[-1,:,:]和h[-1,:,:]相等吗？ 相等

print("out",out)

print("(h,c)",(h,c.size()))
