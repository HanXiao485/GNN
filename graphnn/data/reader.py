# -*- coding: utf-8 -*-
"""数据读取模块：包含 GraphData 类和数据加载函数"""
import csv
import json
import torch

class GraphData:
    """图数据类，存储图的邻接矩阵、节点特征、标签和数据集划分等信息"""
    def __init__(self, adjacency, features, labels=None):
        # 邻接矩阵（Tensor，大小 [N, N]）
        self.adjacency = adjacency
        # 节点特征矩阵（Tensor，大小 [N, F]）
        self.features = features
        # 节点标签（Tensor，大小 [N]），可选
        self.labels = labels
        # 训练/验证/测试集掩码（Tensor，大小 [N]）
        self.train_mask = None
        self.val_mask = None
        self.test_mask = None

    def to(self, device):
        """将数据移动到指定设备"""
        self.adjacency = self.adjacency.to(device)
        self.features = self.features.to(device)
        if self.labels is not None:
            self.labels = self.labels.to(device)
        if self.train_mask is not None:
            self.train_mask = self.train_mask.to(device)
            self.val_mask = self.val_mask.to(device)
            self.test_mask = self.test_mask.to(device)
        return self

def load_graph_from_csv(edge_path, feature_path=None, label_path=None):
    """从CSV文件加载图数据：边列表、节点特征和节点标签（可选）"""
    edges = []
    nodes = set()
    # 读取边列表
    with open(edge_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:  # 跳过空行
                continue
            # 跳过标题行（假设第一列为非数字）
            try:
                float(row[0])
            except:
                continue
            src = row[0]
            tgt = row[1]
            edges.append((src, tgt))
            nodes.add(src)
            nodes.add(tgt)
    # 读取节点特征
    node_features = {}
    feat_dim = 0
    if feature_path:
        with open(feature_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue
                try:
                    float(row[0])
                except:
                    continue
                node_id = row[0]
                feats = [float(x) for x in row[1:]]
                node_features[node_id] = feats
                feat_dim = len(feats)
    # 读取节点标签
    labels = {}
    if label_path:
        with open(label_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue
                try:
                    float(row[0])
                except:
                    continue
                node_id = row[0]
                labels[node_id] = int(row[1])
    # 构建节点索引
    nodes = sorted(nodes)
    node2idx = {node: i for i, node in enumerate(nodes)}
    n = len(nodes)
    # 构建邻接矩阵（无向图）
    adj = torch.zeros((n, n), dtype=torch.float32)
    for src, tgt in edges:
        i = node2idx[src]
        j = node2idx[tgt]
        adj[i, j] = 1.0
        adj[j, i] = 1.0
    # 构建特征矩阵
    if feature_path and node_features:
        features = torch.zeros((n, feat_dim), dtype=torch.float32)
        for node_id, feats in node_features.items():
            idx = node2idx[node_id]
            features[idx] = torch.tensor(feats, dtype=torch.float32)
    else:
        # 如果没有提供节点特征，使用单位向量表示（one-hot）
        features = torch.eye(n, dtype=torch.float32)
    # 构建标签向量
    if label_path and labels:
        label_tensor = torch.zeros(n, dtype=torch.long)
        for node_id, label in labels.items():
            idx = node2idx[node_id]
            label_tensor[idx] = label
    else:
        label_tensor = None
    return GraphData(adj, features, label_tensor)

def load_graph_from_json(json_path):
    """从JSON文件加载图数据，需包含 'nodes' 和 'edges' 字段"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    nodes = []
    features_dict = {}
    labels_dict = {}
    # 解析节点信息
    for node in data.get('nodes', []):
        node_id = node.get('id')
        nodes.append(node_id)
        if 'features' in node:
            features_dict[node_id] = node['features']
        if 'label' in node:
            labels_dict[node_id] = node['label']
    # 解析边信息
    edges = []
    for edge in data.get('edges', []):
        src = edge.get('source')
        tgt = edge.get('target')
        if src is not None and tgt is not None:
            edges.append((src, tgt))
    # 构建节点索引
    nodes = sorted(set(nodes))
    node2idx = {node: i for i, node in enumerate(nodes)}
    n = len(nodes)
    # 构建邻接矩阵
    adj = torch.zeros((n, n), dtype=torch.float32)
    for src, tgt in edges:
        i = node2idx[src]; j = node2idx[tgt]
        adj[i, j] = 1.0; adj[j, i] = 1.0
    # 构建特征矩阵
    if features_dict:
        feat_dim = len(next(iter(features_dict.values())))
        features = torch.zeros((n, feat_dim), dtype=torch.float32)
        for node_id, feats in features_dict.items():
            idx = node2idx[node_id]
            features[idx] = torch.tensor(feats, dtype=torch.float32)
    else:
        features = torch.eye(n, dtype=torch.float32)
    # 构建标签向量
    if labels_dict:
        label_tensor = torch.zeros(n, dtype=torch.long)
        for node_id, label in labels_dict.items():
            idx = node2idx[node_id]
            label_tensor[idx] = label
    else:
        label_tensor = None
    return GraphData(adj, features, label_tensor)
