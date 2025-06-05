import json
import torch
import networkx as nx
from torch.utils.data import DataLoader
from torch_geometric.data import Data, Dataset
import numpy as np

class GraphDataset(Dataset):
    def __init__(self, json_path):
        super().__init__()
        with open(json_path, 'r') as f:
            self.raw_data = json.load(f)

        self.graphs = []
        for entry in self.raw_data:
            keys_sorted = sorted(entry.keys(), key=lambda x: int(x))
            node_keys = keys_sorted[:-1]  # 前面的键是节点
            label_key = keys_sorted[-1]   # 最后一个键是标签

            node_features = []
            for key in node_keys:
                node_features.append(entry[key])  # 每个节点的第一个特征向量
            x = torch.tensor(node_features, dtype=torch.float32)  # 节点特征矩阵 (3, 3)

            x = x.view(x.shape[0], x.shape[1]*x.shape[2]) 

            edge_index = generate_complete_edge_index(len(node_keys))

            # 标签向量（假设为数值向量）
            y = torch.tensor(entry[label_key], dtype=torch.float32)  # 标签向量 (1)

            data = Data(x=x, edge_index=edge_index, y=y)
            self.graphs.append(data)

    def len(self):
        return len(self.graphs)

    def get(self, idx):
        return self.graphs[idx]


def generate_complete_edge_index(n_nodes):
    # 生成无向完全图
    G = nx.complete_graph(n_nodes)
    # 提取边并添加双向连接
    edge_list = []
    for u, v in G.edges():
        edge_list.extend([[u, v], [v, u]])  # 添加双向边
    # 转换为PyTorch张量
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    return edge_index


# 使用示例
if __name__ == "__main__":
    from torch_geometric.loader import DataLoader as GeoDataLoader

    dataset = GraphDataset("node_features.json")  # 替换为你的 JSON 文件路径
    dataloader = GeoDataLoader(dataset, batch_size=1, shuffle=True)

    for batch in dataloader:
        print(batch.y)
        # 访问 batch.x, batch.edge_index, batch.y, batch.batch 等属性
