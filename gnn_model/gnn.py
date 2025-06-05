import torch 
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool

class GNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.fc = torch.nn.Linear(output_dim, 3) 

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = F.leaky_relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)

        # 全局平均池化聚合所有节点特征 [2,5](@ref)
        x = global_mean_pool(x, batch)  # 输出形状 [batch_size, output_dim]
        
        # 如果output_dim≠3，通过线性层调整维度 [2](@ref)
        x = self.fc(x)  

        # return F.log_softmax(x, dim=1)
        return x