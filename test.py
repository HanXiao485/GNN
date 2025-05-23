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
            node_features = []
            for key in sorted(entry.keys(), key=lambda x: int(x)):
                node_features.append(entry[key][0])  # 取每个节点的第一个特征向量
            x = torch.tensor(node_features, dtype=torch.float32)  # 节点特征矩阵 (3, 3)

            # 构造完全图 (3个节点的完全图有6条边)
            edge_index = torch.tensor([
                [0, 0, 1, 1, 2, 2],
                [1, 2, 0, 2, 0, 1]
            ], dtype=torch.long)  # 形状 (2, num_edges)

            data = Data(x=x, edge_index=edge_index)
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
        print(batch)
        # batch 是一个 torch_geometric.data.Batch 对象，支持 .x, .edge_index, .batch 等属性
