import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, global_add_pool, global_sort_pool

class HybridGCN_SAGE(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=3):
        super().__init__()
        # 混合卷积层（GCN捕捉全局结构，SAGE学习邻域聚合）
        self.gcn = GCNConv(input_dim, hidden_dim)
        self.sage = SAGEConv(hidden_dim, hidden_dim)
        # 自适应池化（根据图大小动态选择池化方式）
        self.pool = torch.nn.ModuleDict({
            'small': global_add_pool,       # 节点数<50时使用加法池化
            'large': global_sort_pool       # 节点数≥50时使用排序池化
        })
        # 输出层
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # GCN层
        x_gcn = self.gcn(x, edge_index)
        x_gcn = F.relu(x_gcn)
        
        # SAGE层（使用GCN输出作为输入）
        x = self.sage(x_gcn, edge_index)
        x = F.relu(x)
        
        # 自适应池化（根据图的最大节点数选择策略）
        max_nodes = int(batch[-1]) + 1  # 计算当前batch的最大图节点数
        if max_nodes < 50:
            x = self.pool['small'](x, batch)
        else:
            x = self.pool['large'](x, batch, k=min(10, max_nodes//5))  # 取前10%节点
        
        return self.fc(x)
    