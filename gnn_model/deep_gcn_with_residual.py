import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
from torch_geometric.utils import degree

class DeepGCNWithResidual(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=3):
        super().__init__()
        # 输入层
        self.conv_in = GCNConv(input_dim, hidden_dim)
        # 中间层（带残差连接）
        self.conv_blocks = torch.nn.ModuleList([
            GCNConv(hidden_dim, hidden_dim) 
            for _ in range(3)  # 增加至5层（输入+3中间+输出）
        ])
        # 输出层
        self.conv_out = GCNConv(hidden_dim, hidden_dim)
        # 归一化层
        self.norm = torch.nn.LayerNorm(hidden_dim)
        # 混合池化后的全连接层
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(2*hidden_dim, hidden_dim),  # 平均+最大池化拼接
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # 输入层
        x = self.conv_in(x, edge_index)
        x = F.relu(x)
        x = self.norm(x)
        
        # 残差块
        for conv in self.conv_blocks:
            residual = x
            x = conv(x, edge_index)
            x = F.relu(x)
            x = self.norm(x)
            x += residual  # 残差连接
            x = F.dropout(x, p=0.3, training=self.training)
        
        # 输出层
        x = self.conv_out(x, edge_index)
        
        # 混合池化（平均+最大）
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1)  # 拼接增强特征
        
        return self.fc(x)
    