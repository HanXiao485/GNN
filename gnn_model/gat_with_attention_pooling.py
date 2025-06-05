import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GlobalAttention

class GATWithAttentionPool(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=3):
        super().__init__()
        # 注意力卷积层（头数=4，最后拼接后投影回hidden_dim）
        self.conv1 = GATConv(input_dim, hidden_dim//4, heads=4, dropout=0.2)
        self.conv2 = GATConv(hidden_dim, hidden_dim//4, heads=4, dropout=0.2)
        # 全局注意力池化（带可学习的注意力门）
        self.pool = GlobalAttention(gate_nn=torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim//2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim//2, 1)
        ))
        # 输出层
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        x = self.conv1(x, edge_index)
        x = F.elu(x)  # 更平缓的激活函数
        x = F.dropout(x, p=0.3, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        
        # 全局注意力池化（自动加权重要节点）
        x = self.pool(x, batch)
        
        return self.fc(x)
    