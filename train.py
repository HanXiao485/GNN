from config import GraphDataset
from gnn_model import GNN
import torch.nn.functional as F
from torch_geometric.utils import from_networkx
from torch_geometric.data import Data, DataLoader
from torch_geometric.loader import DataLoader as GeoDataLoader

import matplotlib.pyplot as plt
import torch
import networkx as nx

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# JSON_PATH = "node_features.json"
JSON_PATH = "json_gen/data_v4.json"

dataset = GraphDataset(JSON_PATH)  # 替换为你的 JSON 文件路径
dataloader = GeoDataLoader(dataset, batch_size=16, shuffle=True, drop_last=True)

num_nodes = len(dataloader)  # get number of nodes in the graph

# model initialization
model = GNN(input_dim=16, hidden_dim=4, output_dim=256).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 初始化损失记录列表
train_losses = []

# train loop
model.train()
for epoch in range(2000):

    epoch_loss = 0.0  # 记录当前epoch的总损失
    num_batches = 0   # 记录batch数量

    for batch in dataloader:  # 使用DataLoader迭代
        batch = batch.to(device)
        optimizer.zero_grad()
        output = model(batch)
        loss = F.mse_loss(output, batch.y)  # 注意标签形状可能需要调整（见下方说明）
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()  # 累加每个batch的损失
        num_batches += 1

    # 计算平均损失并记录
    avg_epoch_loss = epoch_loss / num_batches
    train_losses.append(avg_epoch_loss)
    print(f"Epoch: {epoch}, Loss: {loss.item()}")

print("output", output)
print("labels", batch.y)

# loss  curve
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(train_losses)+1), train_losses, 'b-', label='Training Loss')
plt.title('Training Loss Curve', fontsize=14)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('MSE Loss', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.show()