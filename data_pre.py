import numpy as np
from rdkit import Chem
import torch
import pickle
import sys
sys.path.append('..')
num_atom_feat = 34

# 将字符映射到整数以表示 SMILES 字符串中的原子
CHARISOSMISET = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
                 "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
                 "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
                 "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
                 "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
                 "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
                 "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
                 "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}

# 将字符映射到整数以表示蛋白质序列中的氨基酸
CHARPROTSET = {"A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
               "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
               "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
               "U": 19, "T": 20, "W": 21, "V": 22, "Y": 23, "X": 24, "Z": 25}

# 对分类数据进行 one-hot 编码的函数
def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(
            x, allowable_set))
    return [x == s for s in allowable_set]

# 对分类数据进行 one-hot 编码并处理未知输入的函数
def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]

# 提取分子中每个原子的特征
def atom_features(atom,explicit_H=False,use_chirality=True):
    symbol = ['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I', 'other']  #原子符号
    degree = [0, 1, 2, 3, 4, 5, 6]  # 原子度数
    hybridizationType = [Chem.rdchem.HybridizationType.SP,   # 杂化类型
                              Chem.rdchem.HybridizationType.SP2,
                              Chem.rdchem.HybridizationType.SP3,
                              Chem.rdchem.HybridizationType.SP3D,
                              Chem.rdchem.HybridizationType.SP3D2,
                              'other']   # 6-dim
    results = (one_of_k_encoding_unk(atom.GetSymbol(),symbol) + \
                  one_of_k_encoding(atom.GetDegree(),degree) + \
                  [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
                  one_of_k_encoding_unk(atom.GetHybridization(), hybridizationType) + [atom.GetIsAromatic()])  # 编码原子符号+编码原子度数+正电荷和自由电子数+编码杂化+芳香性

    if not explicit_H:
        results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(), # 编码氢的总数
                                                      [0, 1, 2, 3, 4])
    if use_chirality:
        try:
            results = results + one_of_k_encoding_unk(
                    atom.GetProp('_CIPCode'),
                    ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
        except:
            results = results + [False, False] + [atom.HasProp('_ChiralityPossible')]  # 31+3 =34
    return results

# 为分子生成邻接矩阵
def adjacent_matrix(mol):
    adjacency = Chem.GetAdjacencyMatrix(mol)
    return np.array(adjacency)+np.eye(adjacency.shape[0])

# 从 SMILES 字符串中提取特征和邻接矩阵
def mol_features(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
    except:
        raise RuntimeError("SMILES cannot been parsed!")
    atom_feat = np.zeros((mol.GetNumAtoms(), num_atom_feat))
    for atom in mol.GetAtoms():
        atom_feat[atom.GetIdx(), :] = atom_features(atom)
    adj_matrix = adjacent_matrix(mol)
    return atom_feat, adj_matrix

# 从 SMILES 字符串标记特征
def label_smiles(line, smi_ch_ind, MAX_SMI_LEN=100):
    X = np.zeros(MAX_SMI_LEN,dtype=np.int64())
    for i, ch in enumerate(line[:MAX_SMI_LEN]):
        X[i] = smi_ch_ind[ch]
    return X

# 从序列字符串标记特征
def label_sequence(line, smi_ch_ind, MAX_SEQ_LEN=620):
    X = np.zeros(MAX_SEQ_LEN,np.int64())
    for i, ch in enumerate(line[:MAX_SEQ_LEN]):
        X[i] = smi_ch_ind[ch]
    return X



if __name__ == "__main__":

    with open(f"/home/afan/workspace/DTI/DPAG/data/KIBA.txt","r") as f:
        data_list = f.read().strip().split( '\n')

    data_list = [d for d in data_list if '.' not in d.strip().split()[0]]
    N = len(data_list)

    compounds, adjacencies,interactions,smi_ids,prot_ids = [], [], [], [], []

    for no, data in enumerate(data_list):
        drug_id, protein_id, smiles, sequence, interaction = data.strip().split(" ")

        # 标记 SMILES 字符串
        smi_id = torch.from_numpy(label_smiles(smiles, CHARISOSMISET))
        smi_ids.append(torch.LongTensor(smi_id))

        # 标记蛋白质序列
        prot_id = torch.from_numpy(label_sequence(sequence, CHARPROTSET))
        prot_ids.append(torch.LongTensor(prot_id))

        # 提取分子特征和邻接矩阵
        atom_feature, adj = mol_features(smiles)

        label = np.array(interaction, dtype=np.float32)
        atom_feature = torch.FloatTensor(atom_feature)
        adj = torch.FloatTensor(adj)

        label = torch.LongTensor(label)
        compounds.append(atom_feature)
        adjacencies.append(adj)

        interactions.append(label)

    dataset = list(zip(compounds, adjacencies, interactions, smi_ids, prot_ids))
    with open(f"/home/afan/workspace/DTI/DPAG/data/KIBA.pickle", "wb") as f:
        pickle.dump(dataset, f)