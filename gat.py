import torch
import torch.nn as nn
import torch.nn.functional as F
from hyperparameter import hyperparameter

hp = hyperparameter()


class GAT(nn.Module):

    def __init__(self, dropout=0.2):
        super(GAT, self).__init__()
        self.hid_dim = hp.hid_dim  # 隐藏层维度
        self.atom_dim = hp.atom_dim  # 原子特征维度
        self.dropout = dropout  # dropout
        self.do = nn.Dropout(dropout)  # dropout模块

        # GNN 层的线性变换模块列表
        self.W_gnn = nn.ModuleList([nn.Linear(self.atom_dim, self.atom_dim),
                                    nn.Linear(self.atom_dim, self.atom_dim),
                                    nn.Linear(self.atom_dim, self.atom_dim)])

        # GNN 层输出到隐藏层的线性变换模块
        self.W_gnn_trans = nn.Linear(self.atom_dim, self.hid_dim)

        # 化合物注意力机制的参数列表
        self.compound_attn = nn.ParameterList(
            [nn.Parameter(torch.randn(size=(2 * self.atom_dim, 1))) for _ in range(len(self.W_gnn))])

    def forward(self, input, adj):
        for i in range(len(self.W_gnn)):
            h = torch.relu(self.W_gnn[i](input))  # 应用线性变换并激活函数

            size = h.size()[0]
            N = h.size()[1]

            # 计算注意力权重
            a_input = torch.cat([h.repeat(1, 1, N).view(size, N * N, -1), h.repeat(1, N, 1)], dim=2).view(size, N, -1,
                                                                                                          2 * self.atom_dim)  # 拼接输入特征
            e = F.leaky_relu(torch.matmul(a_input, self.compound_attn[i]).squeeze(3))  # 计算注意力得分

            zero_vec = -9e15 * torch.ones_like(e)
            attention = torch.where(adj > 0, e, zero_vec)  # 对注意力得分进行 mask 处理
            attention = F.softmax(attention, dim=2)  # softmax 归一化 (8,41,41)
            attention = F.dropout(attention, self.dropout, training=self.training)  # dropout
            h_prime = torch.matmul(attention, h)  # 应用注意力机制
            input = input + h_prime  # 更新输入特征

        input = self.do(F.relu(self.W_gnn_trans(input)))  # 应用线性变换和激活函数
        return input