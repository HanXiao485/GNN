# -*- coding: utf-8 -*-
"""示例代码：演示如何使用图神经网络库进行数据加载、模型训练和评估"""
import torch
from graphnn.data.reader import load_graph_from_csv, load_graph_from_json
from graphnn.data.preprocessing import normalize_features, normalize_adjacency, train_val_test_split
from graphnn.models.gcn import GCN
from graphnn.models.gat import GAT
from graphnn.models.graphsage import GraphSAGE
from graphnn.train import train_model, test_model

# -------------------------
# 数据加载与预处理
# -------------------------
# 示例1：从CSV加载图数据（边列表、特征、标签）
graph_data = load_graph_from_csv('edges.csv', 'features.csv', 'labels.csv')
# 特征标准化
graph_data.features = normalize_features(graph_data.features)
# 邻接矩阵归一化（适用于 GCN 和 GAT）
graph_data.adjacency = normalize_adjacency(graph_data.adjacency)
# 划分训练/验证/测试集（比例示例：60%/20%/20%）
n = graph_data.features.size(0)
train_mask, val_mask, test_mask = train_val_test_split(
    n, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42)
graph_data.train_mask = train_mask
graph_data.val_mask = val_mask
graph_data.test_mask = test_mask

# 示例2：从JSON加载图数据
# graph_data = load_graph_from_json('graph.json')
# （处理方式同上）

# -------------------------
# 模型定义
# -------------------------
input_dim = graph_data.features.size(1)
hidden_dim = 16
# 假设标签从0到 num_classes-1
output_dim = int(graph_data.labels.max().item() + 1) if graph_data.labels is not None else hidden_dim

# 创建 GCN 模型实例
model = GCN(input_dim, hidden_dim, output_dim, dropout=0.5)

# 如果想使用 GAT 或 GraphSAGE，只需替换模型类
# model = GAT(input_dim, hidden_dim, output_dim, dropout=0.6, alpha=0.2)
# model = GraphSAGE(input_dim, hidden_dim, output_dim, dropout=0.5)

# （可选）将模型和数据移动到GPU
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = model.to(device)
# graph_data = graph_data.to(device)

# -------------------------
# 模型训练
# -------------------------
loss_list, acc_list = train_model(model, graph_data, epochs=100, lr=0.01, weight_decay=5e-4)

# -------------------------
# 模型测试
# -------------------------
test_acc = test_model(model, graph_data)
