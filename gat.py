#!/usr/bin/env Python
# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from hyperparameter import hyperparameter

hp = hyperparameter()

class GAT(nn.Module):

    def __init__(self, dropout = 0.2):
        super(GAT, self).__init__()
        self.hid_dim = hp.hid_dim
        self.atom_dim = hp.atom_dim
        self.dropout = dropout
        self.do = nn.Dropout(dropout)

        self.W_gnn = nn.ModuleList([nn.Linear(self.atom_dim, self.atom_dim),
                                    nn.Linear(self.atom_dim, self.atom_dim),
                                    nn.Linear(self.atom_dim, self.atom_dim)])
        self.W_gnn_trans = nn.Linear(self.atom_dim, self.hid_dim)

        self.compound_attn = nn.ParameterList([nn.Parameter(torch.randn(size=(2 * self.atom_dim, 1))) for _ in range(len(self.W_gnn))])


    def forward(self, input, adj): #input(8,41,34) adj(8,41,41)
        for i in range(len(self.W_gnn)):

            h = torch.relu(self.W_gnn[i](input)) #(8，41，34）

            size = h.size()[0]
            N = h.size()[1]
            a_input = torch.cat([h.repeat(1, 1, N).view(size, N * N, -1), h.repeat(1, N, 1)], dim=2).view(size, N, -1, 2 * self.atom_dim) #(8, 41, 41,68)
            e = F.leaky_relu(torch.matmul(a_input, self.compound_attn[i]).squeeze(3)) #(8, 41, 41)

            zero_vec = -9e15 * torch.ones_like(e)

            attention = torch.where(adj > 0, e, zero_vec)
            attention = F.softmax(attention, dim=2)
            attention = F.dropout(attention, self.dropout, training=self.training) #(8, 41, 41)
            h_prime = torch.matmul(attention, h) #(8, 41, 34)
            input = input + h_prime #(8, 41, 34)

        input = self.do(F.relu(self.W_gnn_trans(input))) #(8, 41, 34)
        return input