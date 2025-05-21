# -*- coding: utf-8 -*-
"""预处理模块：特征归一化、邻接矩阵归一化和数据集划分"""
import torch

def normalize_features(features):
    """对节点特征矩阵按列进行标准化（均值为0，标准差为1）"""
    mean = features.mean(dim=0, keepdim=True)
    std = features.std(dim=0, keepdim=True)
    std[std == 0] = 1  # 防止除以0
    return (features - mean) / std

def normalize_adjacency(adj):
    """对邻接矩阵进行归一化：A_hat = D^{-1/2} (A + I) D^{-1/2}。"""
    # 添加自环
    A = adj + torch.eye(adj.size(0))
    # 计算度矩阵
    deg = A.sum(dim=1)
    # 计算 D^{-1/2}
    deg_inv_sqrt = torch.pow(deg, -0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
    D_inv_sqrt = torch.diag(deg_inv_sqrt)
    # 返回归一化邻接矩阵
    return D_inv_sqrt @ A @ D_inv_sqrt

def train_val_test_split(n_nodes, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=None):
    """将节点索引随机划分为训练/验证/测试集，并返回对应的掩码"""
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError('train/val/test 比例之和必须为1')
    if seed is not None:
        torch.manual_seed(seed)
    indices = torch.randperm(n_nodes)
    n_train = int(train_ratio * n_nodes)
    n_val = int(val_ratio * n_nodes)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    # 构建布尔掩码
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    return train_mask, val_mask, test_mask
