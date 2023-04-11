import random
import os
from model import DPAG
from hyperparameter import hyperparameter
import pickle
from sklearn.model_selection import StratifiedKFold
from transformers import AdamW
from transformers import get_cosine_schedule_with_warmup
import timeit
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, precision_score, recall_score,precision_recall_curve, auc


hp = hyperparameter()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def pack(atoms, adjs,  labels, smi_ids, prot_ids, device):
    atoms_len = 0
    proteins_len = 0
    N = len(atoms)
    atom_num, protein_num = [], []

    for atom in atoms:
        if atom.shape[0] >= atoms_len:
            atoms_len = atom.shape[0]

    if atoms_len>hp.MAX_DRUG_LEN: atoms_len = hp.MAX_DRUG_LEN
    atoms_new = torch.zeros((N,atoms_len,34), device=device)
    i = 0
    for atom in atoms:
        a_len = atom.shape[0]
        if a_len>atoms_len: a_len = atoms_len
        atoms_new[i, :a_len, :] = atom[:a_len, :]
        i += 1
    adjs_new = torch.zeros((N, atoms_len, atoms_len), device=device)
    i = 0
    for adj in adjs:
        a_len = adj.shape[0]
        adj = adj + torch.eye(a_len)
        if a_len>atoms_len: a_len = atoms_len
        adjs_new[i, :a_len, :a_len] = adj[:a_len, :a_len]
        i += 1

    if proteins_len>hp.MAX_PROTEIN_LEN: proteins_len = hp.MAX_PROTEIN_LEN
    proteins_new = torch.zeros((N, proteins_len, 100), device=device)
    i = 0
    labels_new = torch.zeros(N, dtype=torch.long, device=device)
    i = 0
    for label in labels:
        labels_new[i] = label
        i += 1

    smi_id_len = 0
    for smi_id in smi_ids:
        atom_num.append(len(smi_id))
        if len(smi_id) >= smi_id_len:
            smi_id_len = len(smi_id)

    if smi_id_len>hp.MAX_DRUG_LEN: smi_id_len = hp.MAX_DRUG_LEN
    smi_ids_new = torch.zeros([N, smi_id_len], dtype=torch.long, device=device)
    for i, smi_id in enumerate(smi_ids):
        t_len = len(smi_id)
        if t_len>smi_id_len: t_len = smi_id_len
        smi_ids_new[i, :t_len] = smi_id[:t_len]
    prot_id_len = 0
    for prot_id in prot_ids:
        protein_num.append(len(prot_id))
        if len(prot_id) >= prot_id_len: prot_id_len = len(prot_id)

    if prot_id_len>hp.MAX_PROTEIN_LEN: prot_id_len = hp.MAX_PROTEIN_LEN
    prot_ids_new = torch.zeros([N, prot_id_len], dtype=torch.long, device=device)
    for i, prot_id in enumerate(prot_ids):
        t_len = len(prot_id)
        if t_len>prot_id_len: t_len = prot_id_len
        prot_ids_new[i, :t_len] = prot_id[:t_len]
    return (atoms_new, adjs_new, labels_new, smi_ids_new, prot_ids_new, atom_num, protein_num)

class train_model(object):
    def __init__(self, model, lr, weight_decay, batch, n_sample, epochs):
        self.model = model
        weight_p, bias_p = [], []
        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        for name, p in self.model.named_parameters():
            if 'bias' in name:
                bias_p += [p]
            else:
                weight_p += [p]
        self.optimizer = AdamW([{'params': weight_p, 'weight_decay': weight_decay}, {'params': bias_p, 'weight_decay': 0}], lr=lr)
        self.lr_scheduler = get_cosine_schedule_with_warmup(self.optimizer, num_warmup_steps=10, num_training_steps=(n_sample // batch)*epochs)
        self.batch = batch

    def train(self, dataset, device):
        self.model.train()
        np.random.shuffle(dataset)
        N = len(dataset)
        loss_total = 0
        i = 0
        self.optimizer.zero_grad()
        adjs, atoms,  labels, smi_ids, prot_ids = [], [],  [], [], []
        for data in dataset:
            i = i+1
            atom, adj,  label, smi_id, prot_id = data
            adjs.append(adj)
            atoms.append(atom)
            labels.append(label)
            smi_ids.append(smi_id)
            prot_ids.append(prot_id)
            if i % 8 == 0:
                data_pack = pack(atoms, adjs, labels, smi_ids, prot_ids, device)
                loss = self.model(data_pack)
                loss.backward()
                adjs, atoms,  labels, smi_ids, prot_ids = [], [],  [], [], []
            else:
                continue
            if i % self.batch == 0 or i == N:
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
            loss_total += loss.item()
        return loss_total

class test_model(object):
    def __init__(self, model):
        self.model = model

    def test(self, dataset):
        self.model.eval()
        T, Y, S = [], [], []
        with torch.no_grad():
            for data in dataset:
                adjs, atoms,  labels, smi_ids, prot_ids = [],  [], [], [], []
                atom, adj, label, smi_id, prot_id = data
                adjs.append(adj)
                atoms.append(atom)

                labels.append(label)
                smi_ids.append(smi_id)
                prot_ids.append(prot_id)

                data = pack(atoms,adjs, labels, smi_ids, prot_ids, self.model.device)
                correct_labels, predicted_labels, predicted_scores = self.model(data, train=False)
                T.extend(correct_labels)
                Y.extend(predicted_labels)
                S.extend(predicted_scores)
        AUC = roc_auc_score(T, S)
        tpr, fpr, _ = precision_recall_curve(T, S)
        PRC = auc(fpr, tpr)
        precision = precision_score(T, Y)
        recall = recall_score(T, Y)
        return AUC, PRC, precision, recall

    def save_AUCs(self, AUCs, filename):
        with open(filename, 'a') as f:
            f.write('\t'.join(map(str, AUCs)) + '\n')

    def save_model(self, model, filename):
        torch.save(model.state_dict(), filename)


def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset

def split_dataset(dataset, ratio):
    n = int(ratio * len(dataset))
    dataset_1, dataset_2 = dataset[:n], dataset[n:]
    return dataset_1, dataset_2

def init_seed(SEED):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
if __name__ == "__main__":
    init_seed(hp.seed)
    with open(f'./data/{hp.model_name}.pickle',"rb") as f:
        data = pickle.load(f)
    dataset = shuffle_dataset(data, 1234)

    labels = [i[-3] for i in dataset]
    best_epoch, best_AUC_test, best_AUPR_test, best_precision_test, best_recall_test = 0.0, 0.0, 0.0, 0.0, 0.0
    skf = StratifiedKFold(n_splits=hp.n_folds)

    results = np.array([0.0]*4)
    for fold, (train_idx, test_idx) in enumerate(skf.split(dataset, labels)):
        dataset_train = [dataset[idx] for idx in train_idx]
        dataset_test = [dataset[idx] for idx in test_idx]
        dataset_dev, dataset_test = split_dataset(dataset_test, 0.5)

        model = DPAG(hp.dropout, device)


        model.to(device)
        trainer = train_model(model, hp.lr, hp.weight_decay, hp.batch, len(dataset_train), hp.iteration)
        tester = test_model(model)

        if hp.save_name == 'test':
            file_AUCs = f'./result/{hp.model_name}_{fold}.txt'
            file_auc_test = f'./result/test_{hp.model_name}_{fold}.txt'
            file_model = f'./model/{hp.model_name}_{fold}.pt'
        else:
            file_AUCs = f'./result/{hp.save_name}_{fold}.txt'
            file_auc_test = f'./result/test_{hp.save_name}_{fold}.txt'
            file_model = f'./model/{hp.save_name}_{fold}.pt'

        AUCs = ('Epoch\tTime(sec)\tLoss_train\tAUC_dev\tPRC_dev\tPrecison_dev\tRecall_dev')
        with open(file_AUCs, 'w') as f:
            f.write(AUCs + '\n')

        print('Training...')
        print(AUCs)
        start = timeit.default_timer()

        max_AUC_dev = 0
        for epoch in range(1, hp.iteration+1):
            loss_train = trainer.train(dataset_train, device)
            AUC_dev, PRC_dev, PRE_dev, REC_dev  = tester.test(dataset_dev)

            end = timeit.default_timer()
            time = end - start

            AUCs = [epoch, time//60, loss_train, AUC_dev,PRC_dev, PRE_dev, REC_dev]
            tester.save_AUCs(AUCs, file_AUCs)
            if AUC_dev > max_AUC_dev:
                tester.save_model(model, file_model)
                max_AUC_dev = AUC_dev

                test_auc, test_prc, test_pre, test_recall = tester.test(dataset_test)
                tester.save_AUCs([epoch, test_auc, test_prc, test_pre, test_recall], file_auc_test)
                best_epoch = epoch
                best_AUC_test = test_auc
                best_AUPR_test = test_prc
                best_precision_test = test_pre
                best_recall_test = test_recall
                print(f'Test ---> AUC: {test_auc}, PRC: {test_prc}')
            print('\t'.join(map(str, AUCs)))

        results += np.array([best_AUC_test, best_AUPR_test, best_precision_test, best_recall_test])
    results /= hp.n_folds
    print('\t'.join(map(str, results)) + '\n')