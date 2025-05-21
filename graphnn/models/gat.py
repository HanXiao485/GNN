# -*- coding: utf-8 -*-
"""GAT模型模块：包括GraphAttentionLayer层和GAT模型类"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphAttentionLayer(nn.Module):
    """单头图注意力层（GAT）"""
    def __init__(self, in_features, out_features, dropout=0.6, alpha=0.2, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha  # LeakyReLU 的负斜率
        self.concat = concat

        # 定义权重矩阵 W 和注意力向量 a
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        self.reset_parameters()
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def reset_parameters(self):
        # Xavier 初始化
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W)  # [N, out_features]
        N = Wh.size()[0]
        # 将 Wh_i 和 Wh_j 拼接以计算注意力系数
        a_input = torch.cat([Wh.repeat(1, N).view(N * N, -1), Wh.repeat(N, 1)], dim=1)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(1)).view(N, N)
        # 对不存在边的位置设置为 -inf，使 softmax 后概率为0
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, p=self.dropout, training=self.training)
        # 计算线性组合
        h_prime = torch.matmul(attention, Wh)  # [N, out_features]
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

class GAT(nn.Module):
    """GAT模型：包含两层GraphAttentionLayer"""
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.6, alpha=0.2):
        super(GAT, self).__init__()
        self.attention1 = GraphAttentionLayer(input_dim, hidden_dim, dropout, alpha, concat=True)
        self.attention2 = GraphAttentionLayer(hidden_dim, output_dim, dropout, alpha, concat=False)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.attention1(x, adj)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.attention2(x, adj)
        return x
