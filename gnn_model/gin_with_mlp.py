import torch
import torch.nn.functional as F
from torch_geometric.nn import GINConv, Set2Set

class GINWithSet2Set(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=3, num_layers=4):
        super().__init__()
        # GIN卷积层（使用MLP作为聚合函数）
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers):
            mlp = torch.nn.Sequential(
                torch.nn.Linear(hidden_dim if i>0 else input_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, hidden_dim)
            )
            self.convs.append(GINConv(mlp))
        
        # Set2Set池化（适合回归任务的序列到序列聚合）
        self.pool = Set2Set(hidden_dim, processing_steps=3)
        # 输出层
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(2*hidden_dim, hidden_dim),  # Set2Set输出维度2*hidden_dim
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.3, training=self.training)
        
        # Set2Set池化（捕捉全局结构信息）
        x = self.pool(x, batch)
        
        return self.fc(x)
    