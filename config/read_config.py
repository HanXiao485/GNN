import json
import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Data, Dataset

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
                node_features.append(entry[key][0])  # 每个节点的第一个特征向量
            x = torch.tensor(node_features, dtype=torch.float32)  # 节点特征矩阵 (3, 3)

            # 构造完全图 (3个节点的完全图有6条边)
            edge_index = torch.tensor([
                [0, 0, 1, 1, 2, 2],
                [1, 2, 0, 2, 0, 1]
            ], dtype=torch.long)  # 形状 (2, num_edges)

            # 标签向量（假设为数值向量）
            y = torch.tensor(entry[label_key][0], dtype=torch.float32)  # 标签向量 (1)

            data = Data(x=x, edge_index=edge_index, y=y)
            self.graphs.append(data)

    def len(self):
        return len(self.graphs)

    def get(self, idx):
        return self.graphs[idx]

# 使用示例
if __name__ == "__main__":
    from torch_geometric.loader import DataLoader as GeoDataLoader

    dataset = GraphDataset("node_features.json")  # 替换为你的 JSON 文件路径
    dataloader = GeoDataLoader(dataset, batch_size=1, shuffle=True)

    for batch in dataloader:
        print(batch.y)
        # 访问 batch.x, batch.edge_index, batch.y, batch.batch 等属性
