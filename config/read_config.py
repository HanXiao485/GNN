# read json file

import torch
import json
import networkx as nx
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import from_networkx

def load_json(json_path):
    with open(json_path, 'r') as f:
        # 读取JSON并转换为张量字典
        data_list = json.load(f)
        features = {}
        all_samples = []
        for data in data_list:
            # 将每个样本字典转换为张量字典
            features = {}
            for k, v in data.items():
                features[int(k)] = torch.tensor(v, dtype=torch.float)
            
            # 验证当前样本的特征形状一致性
            sample_shapes = {tuple(feat.shape) for feat in features.values()}
            if len(sample_shapes) > 1:
                raise ValueError("特征形状不一致")
            feature_shape = sample_shapes.pop()
            
            # 创建当前样本的张量
            num_nodes = len(features)
            x = torch.zeros((num_nodes, *feature_shape), dtype=torch.float)
            for node_id, feat in features.items():
                x[node_id] = feat
            
            all_samples.append(x)

            # 检查所有样本的形状是否相同
            if not all_samples:
                return torch.tensor([])  # 处理空列表的情况
            
            expected_shape = all_samples[0].shape
            for idx, x in enumerate(all_samples[1:], 1):
                if x.shape != expected_shape:
                    raise ValueError(f"样本{idx}的形状与第一个样本不一致")
                
        num_nodes = len(data)    
        G = nx.complete_graph(num_nodes)
        data = from_networkx(G)  # convert networkx graph to pytorch geometric graph (undirected)

        edge_index = data.edge_index.contiguous()
                
        datas = Data(x=torch.stack(all_samples, dim=0), 
                    edge_index=edge_index,
                    y = torch.tensor([[0, 1, 0],[1, 0, 0],[0, 0, 1]]))  # labelsgraph data
    
    return datas