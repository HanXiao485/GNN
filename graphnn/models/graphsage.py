# -*- coding: utf-8 -*-
"""GraphSAGE模型模块：包括GraphSageLayer层和GraphSAGE模型类"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphSageLayer(nn.Module):
    """GraphSAGE层：使用节点自身和邻居均值特征进行聚合"""
    def __init__(self, in_features, out_features, bias=True):
        super(GraphSageLayer, self).__init__()
        # 定义全连接层，输入为自身特征和邻居均值特征拼接
        self.fc = nn.Linear(in_features * 2, out_features, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        # 初始化全连接层参数
        nn.init.xavier_uniform_(self.fc.weight)
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)

    def forward(self, x, adj):
        # 邻居特征求和
        neighbor_sum = torch.matmul(adj, x)  # [N, in_features]
        # 节点度（邻居数量，包括自身）
        deg = adj.sum(1).unsqueeze(1)  # [N, 1]
        deg[deg == 0] = 1
        neighbor_mean = neighbor_sum / deg  # 邻居均值特征
        # 特征拼接（自身特征 + 邻居均值特征）
        h = torch.cat([x, neighbor_mean], dim=1)  # [N, 2*in_features]
        h = self.fc(h)
        return F.relu(h)

class GraphSAGE(nn.Module):
    """GraphSAGE模型：包含两层GraphSageLayer"""
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5):
        super(GraphSAGE, self).__init__()
        self.layer1 = GraphSageLayer(input_dim, hidden_dim)
        self.layer2 = GraphSageLayer(hidden_dim, output_dim)
        self.dropout = dropout

    def forward(self, x, adj):
        # 第1层
        x = self.layer1(x, adj)
        x = F.dropout(x, p=self.dropout, training=self.training)
        # 第2层
        x = self.layer2(x, adj)
        return x
