from config import read_config
from gnn_model import GNN
import torch.nn.functional as F
from torch_geometric.utils import from_networkx
from torch_geometric.data import Data, DataLoader

import torch
import networkx as nx

JSON_PATH = "node_features.json"

features = read_config.load_json(JSON_PATH)  # data is a dict

num_nodes = len(features)  # get number of nodes in the graph

# G = nx.complete_graph(num_nodes)
# data = from_networkx(G)  # convert networkx graph to pytorch geometric graph (undirected)

# edge_index = data.edge_index.contiguous()

# data = Data(x=features, 
#             edge_index=edge_index,
#             y = torch.tensor([[0, 1, 0],[1, 0, 0],[0, 0, 1]]))  # labelsgraph data

loader = DataLoader([features], batch_size=1, shuffle=True)


# model initialization
model = GNN(input_dim=num_nodes, hidden_dim=16, output_dim=3)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


# train loop
model.train()
for epoch in range(1000):
    for batch in loader:  # 使用DataLoader迭代
        optimizer.zero_grad()
        output = model(batch)
        loss = F.nll_loss(output, batch.y)  # 注意标签形状可能需要调整（见下方说明）
        loss.backward()
        optimizer.step()
    print(f"Epoch: {epoch}, Loss: {loss.item()}")