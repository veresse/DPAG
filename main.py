# 导入必要的库
import random
import os
from model import DPAG  # 导入模型
from hyperparameter import hyperparameter  # 导入超参数
import pickle
from sklearn.model_selection import StratifiedKFold
from transformers import AdamW
from transformers import get_cosine_schedule_with_warmup
import timeit
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, precision_score, recall_score, precision_recall_curve, auc

# 实例化超参数对象
hp = hyperparameter()
# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.chdir(os.path.dirname(os.path.abspath(__file__)))  # 设置当前工作目录为文件所在目录


# 定义打包函数，将数据打包成模型可接受的形式
def pack(atoms, adjs, labels, smi_ids, prot_ids, device):
    # 初始化原子和蛋白质序列的长度
    atoms_len = 0
    proteins_len = 0
    # 获取数据集大小
    N = len(atoms)
    atom_num, protein_num = [], []

    # 循环遍历每个分子的原子序列，找出最大的原子序列长度
    for atom in atoms:
        if atom.shape[0] >= atoms_len:
            atoms_len = atom.shape[0]

    # 如果原子序列长度超过了设定的最大长度，则截断
    if atoms_len > hp.MAX_DRUG_LEN:
        atoms_len = hp.MAX_DRUG_LEN
    # 创建新的原子序列张量
    atoms_new = torch.zeros((N, atoms_len, 34), device=device)
    i = 0
    # 将原子序列填充到新的张量中
    for atom in atoms:
        a_len = atom.shape[0]
        if a_len > atoms_len:
            a_len = atoms_len
        atoms_new[i, :a_len, :] = atom[:a_len, :]
        i += 1

    # 创建新的邻接矩阵张量
    adjs_new = torch.zeros((N, atoms_len, atoms_len), device=device)
    i = 0
    # 将邻接矩阵填充到新的张量中
    for adj in adjs:
        a_len = adj.shape[0]
        adj = adj + torch.eye(a_len)
        if a_len > atoms_len:
            a_len = atoms_len
        adjs_new[i, :a_len, :a_len] = adj[:a_len, :a_len]
        i += 1

    # 如果蛋白质序列长度超过了设定的最大长度，则截断
    if proteins_len > hp.MAX_PROTEIN_LEN:
        proteins_len = hp.MAX_PROTEIN_LEN
    # 创建新的蛋白质序列张量
    proteins_new = torch.zeros((N, proteins_len, 100), device=device)
    i = 0
    # 创建新的标签张量
    labels_new = torch.zeros(N, dtype=torch.long, device=device)
    i = 0
    # 将标签填充到新的张量中
    for label in labels:
        labels_new[i] = label
        i += 1

    # 计算SMILES ID的最大长度
    smi_id_len = 0
    for smi_id in smi_ids:
        atom_num.append(len(smi_id))
        if len(smi_id) >= smi_id_len:
            smi_id_len = len(smi_id)

    # 如果SMILES ID的长度超过了设定的最大长度，则截断
    if smi_id_len > hp.MAX_DRUG_LEN:
        smi_id_len = hp.MAX_DRUG_LEN
    # 创建新的SMILES ID张量
    smi_ids_new = torch.zeros([N, smi_id_len], dtype=torch.long, device=device)
    # 将SMILES ID填充到新的张量中
    for i, smi_id in enumerate(smi_ids):
        t_len = len(smi_id)
        if t_len > smi_id_len:
            t_len = smi_id_len
        smi_ids_new[i, :t_len] = smi_id[:t_len]

    # 计算蛋白质ID的最大长度
    prot_id_len = 0
    for prot_id in prot_ids:
        protein_num.append(len(prot_id))
        if len(prot_id) >= prot_id_len:
            prot_id_len = len(prot_id)

    # 如果蛋白质ID的长度超过了设定的最大长度，则截断
    if prot_id_len > hp.MAX_PROTEIN_LEN:
        prot_id_len = hp.MAX_PROTEIN_LEN
    # 创建新的蛋白质ID张量
    prot_ids_new = torch.zeros([N, prot_id_len], dtype=torch.long, device=device)
    # 将蛋白质ID填充到新的张量中
    for i, prot_id in enumerate(prot_ids):
        t_len = len(prot_id)
        if t_len > prot_id_len:
            t_len = prot_id_len
        prot_ids_new[i, :t_len] = prot_id[:t_len]

    # 返回打包后的数据
    return (atoms_new, adjs_new, labels_new, smi_ids_new, prot_ids_new, atom_num, protein_num)


# 定义训练模型类
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
        self.optimizer = AdamW(
            [{'params': weight_p, 'weight_decay': weight_decay}, {'params': bias_p, 'weight_decay': 0}], lr=lr)
        self.lr_scheduler = get_cosine_schedule_with_warmup(self.optimizer, num_warmup_steps=10,
                                                            num_training_steps=(n_sample // batch) * epochs)
        self.batch = batch

    # 训练函数
    def train(self, dataset, device):
        self.model.train()  # 将模型设置为训练模式
        np.random.shuffle(dataset)  # 打乱数据集
        N = len(dataset)  # 数据集大小
        loss_total = 0  # 初始化总损失
        i = 0  # 初始化迭代次数计数器
        self.optimizer.zero_grad()  # 梯度清零
        adjs, atoms, labels, smi_ids, prot_ids = [], [], [], [], []  # 初始化存储数据的列表
        for data in dataset:
            i = i + 1  # 迭代次数加一
            atom, adj, label, smi_id, prot_id = data  # 从数据中获取原子序列、邻接矩阵、标签、SMILES ID和蛋白质ID
            adjs.append(adj)  # 将邻接矩阵添加到列表中
            atoms.append(atom)  # 将原子序列添加到列表中
            labels.append(label)  # 将标签添加到列表中
            smi_ids.append(smi_id)  # 将SMILES ID添加到列表中
            prot_ids.append(prot_id)  # 将蛋白质ID添加到列表中
            if i % 8 == 0:  # 每处理8个数据执行一次
                data_pack = pack(atoms, adjs, labels, smi_ids, prot_ids, device)  # 数据打包
                loss = self.model(data_pack)  # 前向传播计算损失
                loss.backward()  # 反向传播计算梯度
                adjs, atoms, labels, smi_ids, prot_ids = [], [], [], [], []  # 清空存储数据的列表
            else:
                continue
            if i % self.batch == 0 or i == N:  # 每处理一个batch或者处理完所有数据时执行
                self.optimizer.step()  # 更新模型参数
                self.lr_scheduler.step()  # 更新学习率
                self.optimizer.zero_grad()  # 梯度清零
            loss_total += loss.item()  # 累加损失
        return loss_total  # 返回总损失


# 定义测试模型类
class test_model(object):
    def __init__(self, model):
        self.model = model

    # 测试函数
    def test(self, dataset):
        self.model.eval()  # 将模型设置为评估模式
        T, Y, S = [], [], []  # 初始化真实标签、预测标签和预测分数列表
        with torch.no_grad():
            for data in dataset:
                adjs, atoms, labels, smi_ids, prot_ids = [], [], [], [], []  # 初始化存储数据的列表
                atom, adj, label, smi_id, prot_id = data  # 从数据中获取原子序列、邻接矩阵、标签、SMILES ID和蛋白质ID
                adjs.append(adj)  # 将邻接矩阵添加到列表中
                atoms.append(atom)  # 将原子序列添加到列表中
                labels.append(label)  # 将标签添加到列表中
                smi_ids.append(smi_id)  # 将SMILES ID添加到列表中
                prot_ids.append(prot_id)  # 将蛋白质ID添加到列表中
                data = pack(atoms, adjs, labels, smi_ids, prot_ids, self.model.device)  # 数据打包
                # 调用模型进行预测
                correct_labels, predicted_labels, predicted_scores = self.model(data, train=False)
                # 将预测结果添加到相应的列表中
                T.extend(correct_labels)
                Y.extend(predicted_labels)
                S.extend(predicted_scores)
        # 计算AUC和PRC
        AUC = roc_auc_score(T, S)
        tpr, fpr, _ = precision_recall_curve(T, S)
        PRC = auc(fpr, tpr)
        # 计算准确率和召回率
        precision = precision_score(T, Y)
        recall = recall_score(T, Y)
        return AUC, PRC, precision, recall  # 返回评估指标值

    # 保存AUC到文件
    def save_AUCs(self, AUCs, filename):
        with open(filename, 'a') as f:
            f.write('\t'.join(map(str, AUCs)) + '\n')

    # 保存模型到文件
    def save_model(self, model, filename):
        torch.save(model.state_dict(), filename)


# 打乱数据集
def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset


# 将数据集分割成训练集和测试集
def split_dataset(dataset, ratio):
    n = int(ratio * len(dataset))
    dataset_1, dataset_2 = dataset[:n], dataset[n:]
    return dataset_1, dataset_2


# 初始化随机种子
def init_seed(SEED):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True


# 设置可见的GPU设备
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == "__main__":
    init_seed(hp.seed)  # 初始化随机种子
    with open(f'./data/{hp.model_name}.pickle', "rb") as f:
        data = pickle.load(f)  # 从文件中加载数据集
    dataset = shuffle_dataset(data, 1234)  # 打乱数据集

    labels = [i[-3] for i in dataset]  # 获取标签列表
    best_epoch, best_AUC_test, best_AUPR_test, best_precision_test, best_recall_test = 0.0, 0.0, 0.0, 0.0, 0.0  # 初始化最佳结果指标
    skf = StratifiedKFold(n_splits=hp.n_folds)  # 初始化交叉验证对象

    results = np.array([0.0] * 4)  # 初始化结果数组
    for fold, (train_idx, test_idx) in enumerate(skf.split(dataset, labels)):
        dataset_train = [dataset[idx] for idx in train_idx]  # 获取训练集数据
        dataset_test = [dataset[idx] for idx in test_idx]  # 获取测试集数据
        dataset_dev, dataset_test = split_dataset(dataset_test, 0.5)  # 将测试集分割成验证集和测试集

        model = DPAG(hp.dropout, device)  # 初始化模型对象

        model.to(device)  # 将模型移动到GPU上
        trainer = train_model(model, hp.lr, hp.weight_decay, hp.batch, len(dataset_train), hp.iteration)  # 初始化训练器
        tester = test_model(model)  # 初始化测试器

        if hp.save_name == 'test':
            file_AUCs = f'./result/{hp.model_name}_{fold}.txt'  # 定义AUC文件名
            file_auc_test = f'./result/test_{hp.model_name}_{fold}.txt'  # 定义测试AUC文件名
            file_model = f'./model/{hp.model_name}_{fold}.pt'  # 定义模型文件名
        else:
            file_AUCs = f'./result/{hp.save_name}_{fold}.txt'  # 定义AUC文件名
            file_auc_test = f'./result/test_{hp.save_name}_{fold}.txt'  # 定义测试AUC文件名
            file_model = f'./model/{hp.save_name}_{fold}.pt'  # 定义模型文件名

        AUCs = ('Epoch\tTime(sec)\tLoss_train\tAUC_dev\tPRC_dev\tPrecison_dev\tRecall_dev')  # 定义AUC字符串
        with open(file_AUCs, 'w') as f:
            f.write(AUCs + '\n')  # 写入AUC字符串

        print('Training...')  # 输出训练提示
        print(AUCs)  # 输出AUC字符串
        start = timeit.default_timer()  # 开始计时

        max_AUC_dev = 0  # 初始化最大AUC
        for epoch in range(1, hp.iteration + 1):  # 迭代训练
            loss_train = trainer.train(dataset_train, device)  # 训练模型
            AUC_dev, PRC_dev, PRE_dev, REC_dev = tester.test(dataset_dev)  # 测试模型性能

            end = timeit.default_timer()  # 计时结束
            time = end - start  # 计算耗时

            AUCs = [epoch, time // 60, loss_train, AUC_dev, PRC_dev, PRE_dev, REC_dev]  # 计算结果指标
            tester.save_AUCs(AUCs, file_AUCs)  # 保存AUC到文件
            if AUC_dev > max_AUC_dev:
                tester.save_model(model, file_model)  # 保存模型到文件
                max_AUC_dev = AUC_dev  # 更新最大AUC

                test_auc, test_prc, test_pre, test_recall = tester.test(dataset_test)  # 测试模型性能
                tester.save_AUCs([epoch, test_auc, test_prc, test_pre, test_recall], file_auc_test)  # 保存测试AUC到文件
                best_epoch = epoch  # 更新最佳迭代次数
                best_AUC_test = test_auc  # 更新最佳测试AUC
                best_AUPR_test = test_prc  # 更新最佳测试PRC
                best_precision_test = test_pre  # 更新最佳测试准确率
                best_recall_test = test_recall  # 更新最佳测试召回率
                print(f'Test ---> AUC: {test_auc}, PRC: {test_prc}')  # 输出测试结果
            print('\t'.join(map(str, AUCs)))  # 输出结果指标

        results += np.array([best_AUC_test, best_AUPR_test, best_precision_test, best_recall_test])  # 更新总结果
    results /= hp.n_folds  # 计算平均结果
    print('\t'.join(map(str, results)) + '\n')  # 输出平均结果
