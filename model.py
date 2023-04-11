"""
This code refers to https://github.com/zhaoqichang/HpyerAttentionDTI and https://github.com/czjczj/IIFDTI.
"""

import torch
import torch.nn.functional as F
import sys
from torch import nn
import numpy as np
from gat import GAT
from transformer import transformer
from hyperparameter import hyperparameter

hp = hyperparameter()

sys.path.append('..')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DPAG(nn.Module):
    def __init__(self, dropout, device):
        super(DPAG, self).__init__()
        

        self.batch_size = hp.batch
        self.max_drug = hp.MAX_DRUG_LEN
        self.max_protein = hp.MAX_PROTEIN_LEN
        self.dropout = dropout
        self.device = device
        self.hid_dim = hp.hid_dim
        self.pro_em = hp.pro_em
        self.smi_em = hp.smi_em
        self.prot_embed = nn.Embedding(self.pro_em, self.hid_dim, padding_idx=0)
        self.smi_embed = nn.Embedding(self.smi_em, self.hid_dim, padding_idx=0)
        self.smi_tf = transformer(self.hid_dim,self.hid_dim)
        self.pro_tf = transformer(self.hid_dim,self.hid_dim)
        self.gat = GAT()
        self.sigmoid = nn.Sigmoid()
        self.att_layer = torch.nn.Linear(64, 64)

        self.relu = nn.ReLU()

        self.out = nn.Sequential(
            nn.Linear(128, 1024), 
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512,2)
        )
        self.do = nn.Dropout(dropout)


    def forward(self, compound, adj, smi_ids, prot_ids):
        drug_gat = self.gat(compound, adj)

        prot_embed = self.prot_embed(prot_ids)
        smi_embed = self.smi_embed(smi_ids)

        smi_tf = self.smi_tf(smi_embed) 
        pro_tf = self.pro_tf(prot_embed)


        smi_att = self.att_layer(smi_tf)
        pro_att = self.att_layer(pro_tf)
        
        smi_attss = torch.cat([smi_att,drug_gat], dim=1)

        protein = torch.unsqueeze(pro_att, 1).repeat(1, smi_attss.shape[-2],1, 1)
        drug = torch.unsqueeze(smi_attss, 2).repeat(1, 1, pro_tf.shape[-2], 1) 
        Atten_matrix = self.att_layer(self.relu(protein + drug))
        smi_atts = self.sigmoid(torch.mean(Atten_matrix, 2))
        pro_atts = self.sigmoid(torch.mean(Atten_matrix, 1))
        
        smi_tfs = 0.5*smi_attss + smi_attss * smi_atts
        pro_tfs = 0.5*pro_tf + pro_tf * pro_atts

        smi = smi_tfs.mean(dim=1)
        pro = pro_tfs.mean(dim=1)

        out_fc = torch.cat([smi,pro], dim=-1)

        predict = self.out(out_fc)

        return predict


    def __call__(self, data, train=True):
        compound, adj, correct_interaction, smi_ids, prot_ids, atom_num, protein_num = data
        weight_ce = torch.FloatTensor([1, 3]).cuda()
        Loss = nn.CrossEntropyLoss(weight=weight_ce)
        if train:
            predicted_interaction = self.forward(compound, adj,  smi_ids, prot_ids)
            loss = Loss(predicted_interaction, correct_interaction)
            return loss
        else:
            predicted_interaction = self.forward(compound, adj,  smi_ids, prot_ids)
            correct_labels = correct_interaction.to('cpu').data.numpy()
            ys = F.softmax(predicted_interaction, 1).to('cpu').data.numpy()
            predicted_labels = np.argmax(ys, axis=1)
            predicted_scores = ys[:, 1]
            return correct_labels, predicted_labels, predicted_scores
