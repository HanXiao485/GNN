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
            self.raw_data = flatten_outer_dict(self.raw_data)

        self.graphs = []
        for entry in self.raw_data:
            keys_sorted = sorted(entry.keys(), key=lambda x: int(x))
            node_keys = keys_sorted[:-1]  # 前面的键是节点
            label_key = keys_sorted[-1]   # 最后一个键是标签

            node_feature = []
            node_features = []
            for key in node_keys:
                node_feature = [item for sublist in entry[key] for item in sublist]
                node_features.append(node_feature)
                # node_features.append(entry[key])  # 每个节点的第一个特征向量
            x = torch.tensor(node_features, dtype=torch.float32)  # 节点特征矩阵 (3, 3)

            # # 构造完全图 (3个节点的完全图有6条边)
            # edge_index = torch.tensor([
            #     [0, 0, 1, 1, 2, 2],
            #     [1, 2, 0, 2, 0, 1]
            # ], dtype=torch.long)  # 形状 (2, num_edges)
            edge_index = generate_complete_edge_index(len(node_keys))

            label_list = entry[label_key]
            if isinstance(label_list[0], list):  # 检查是否嵌套
                label_list = label_list[0]  # 展平为[0,1,0]
            # 标签向量（假设为数值向量）
            y = torch.tensor(label_list, dtype=torch.float32).unsqueeze(0)  # 标签向量 (1)

            data = Data(x=x, edge_index=edge_index, y=y)
            self.graphs.append(data)

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
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

def flatten_outer_dict(outer_dict):
    """
    将外层字典展开为内层字典的列表
    
    参数:
        outer_dict (dict): 外层字典，每个键对应的值应为字典类型
    
    返回:
        list: 包含所有内层字典的列表
    
    异常:
        ValueError: 如果外层字典中存在非字典类型的键值
    """
    inner_dicts = []
    for key, value in outer_dict[0].items():
        if not isinstance(value, dict):
            raise ValueError(f"键 '{key}' 对应的值不是字典类型")
        inner_dicts.append(value)
    return inner_dicts

# 使用示例
if __name__ == "__main__":
    from torch_geometric.loader import DataLoader as GeoDataLoader

    dataset = GraphDataset("node_features.json")  # 替换为你的 JSON 文件路径
    dataloader = GeoDataLoader(dataset, batch_size=1, shuffle=True)

    for batch in dataloader:
        print(batch.y)
        # 访问 batch.x, batch.edge_index, batch.y, batch.batch 等属性
