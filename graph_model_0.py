import json
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.utils import from_networkx
import networkx  as nx

class JSONGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)  # 第一层图卷积[2](@ref)
        self.conv2 = GCNConv(hidden_dim, output_dim)  # 第二层图卷积[1](@ref)
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))  # 非线性激活[7](@ref)
        x = F.dropout(x, p=0.5, training=self.training)  # 正则化[2](@ref)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)  # 分类输出[1](@ref)

def load_graph_from_json(json_path, edge_list):
    """从JSON加载图数据（含特征维度自动检测）[6](@ref)"""
    with open(json_path, 'r') as f:
        features = {int(k): v for k, v in json.load(f).items()}  # 键转换为整数
    
    # 特征维度一致性验证
    dims = set(len(v) for v in features.values())
    if len(dims) > 1:
        raise ValueError("特征维度不一致")
    
    # 构建特征矩阵
    num_nodes = len(features)
    feature_dim = dims.pop()
    x = torch.zeros((num_nodes, feature_dim), dtype=torch.float)
    for node_id, feat in features.items():
        x[node_id] = torch.tensor(feat)
    
    # 构建边索引（支持无向图）[1](@ref)
    # edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_index = edge_list
    return Data(x=x, edge_index=edge_index)

# 测试配置
# EDGES = from_networkx(nx.complete_graph(3)).edge_index  # 完全图结构[7](@ref)环形连接结构[7](@ref)
EDGES = torch.tensor([[0,1],[0,2],[1,2],[2,0]], dtype=torch.long).t().contiguous()
JSON_PATH = "node_features.json"
print(EDGES)

# 加载数据
data = load_graph_from_json(JSON_PATH, EDGES)
data.y = torch.tensor([0, 1, 0])  # 模拟标签（3节点2分类）[1](@ref)

# 初始化模型
model = JSONGNN(input_dim=3, hidden_dim=16, output_dim=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 训练循环
model.train()
for epoch in range(500):
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out, data.y)  # 负对数似然损失[2](@ref)
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch+1:03d} | Loss: {loss.item():.4f}')