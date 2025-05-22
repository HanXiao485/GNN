import networkx as nx
import torch 
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.utils import from_networkx

x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

n = 3
G = nx.complete_graph(n)  # 一键生成全连接无向图

data = from_networkx(G)
print(data.edge_index)

# data = Data(x=x, edge_index=G.edges())
print(data.is_undirected())  # 输出：True，验证是否为无向图

# print("边列表:", G.edges())  # 输出所有无向边
# nx.draw(G, with_labels=True)  # 可视化图结构