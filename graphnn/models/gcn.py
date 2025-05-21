# -*- coding: utf-8 -*-
"""GCN模型模块：包括GraphConvolution层和GCN模型类"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConvolution(nn.Module):
    """简单的图卷积层，参照 Kipf & Welling (2017)"""
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # Xavier 初始化权重
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, input, adj):
        # Graph Convolution: X -> A_hat X W
        support = torch.matmul(input, self.weight)  # XW
        output = torch.matmul(adj, support)        # A_hat XW
        if self.bias is not None:
            output = output + self.bias
        return output

class GCN(nn.Module):
    """GCN模型：由两层图卷积组成"""
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.5):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(input_dim, hidden_dim)
        self.gc2 = GraphConvolution(hidden_dim, output_dim)
        self.dropout = dropout

    def forward(self, x, adj):
        # 第1层卷积 + ReLU + Dropout
        x = self.gc1(x, adj)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        # 第2层卷积
        x = self.gc2(x, adj)
        return x
