from config import GraphDataset
from gnn_model import GNN
import torch.nn.functional as F
from torch_geometric.utils import from_networkx
from torch_geometric.data import Data, DataLoader
from torch_geometric.loader import DataLoader as GeoDataLoader

import matplotlib.pyplot as plt
import torch
import networkx as nx

JSON_PATH = "node_features.json"

dataset = GraphDataset(JSON_PATH)  # 替换为你的 JSON 文件路径
dataloader = GeoDataLoader(dataset, batch_size=1, shuffle=True)

num_nodes = len(dataloader)  # get number of nodes in the graph

# model initialization
model = GNN(input_dim=3, hidden_dim=16, output_dim=3)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 初始化损失记录列表
train_losses = []

# train loop
model.train()
for epoch in range(1000):

    epoch_loss = 0.0  # 记录当前epoch的总损失
    num_batches = 0   # 记录batch数量

    for batch in dataloader:  # 使用DataLoader迭代
        optimizer.zero_grad()
        output = model(batch)
        loss = F.mse_loss(output, batch.y)  # 注意标签形状可能需要调整（见下方说明）
        # print("output:" , output)
        # print("batch.y:" , batch.y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()  # 累加每个batch的损失
        num_batches += 1

    # 计算平均损失并记录
    avg_epoch_loss = epoch_loss / num_batches
    train_losses.append(avg_epoch_loss)
    print(f"Epoch: {epoch}, Loss: {loss.item()}")



# 训练完成后绘制损失曲线
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(train_losses)+1), train_losses, 'b-', label='Training Loss')
plt.title('Training Loss Curve', fontsize=14)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('MSE Loss', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.show()