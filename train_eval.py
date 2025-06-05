from config import GraphDataset
from gnn_model import GNN, HybridGCN_SAGE, DeepGCNWithResidual, GATWithAttentionPool, GINWithSet2Set
from data_process import TargetScaler
import torch.nn.functional as F
from torch_geometric.utils import from_networkx
from torch_geometric.data import Data, DataLoader
from torch_geometric.loader import DataLoader as GeoDataLoader
from torch.utils.data import random_split  # 数据集划分

import matplotlib.pyplot as plt
import torch
import networkx as nx

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

JSON_PATH = "/home/data/xiaohan/datasets/m2f-gnn/result.json"

# 加载数据集并划分训练/验证集（8:2比例）
dataset = GraphDataset(JSON_PATH)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# 创建训练和验证的数据加载器
train_loader = GeoDataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = GeoDataLoader(val_dataset, batch_size=128, shuffle=False)  # 验证集不shuffle

# model initialization
# model = GNN(input_dim=371, hidden_dim=16, output_dim=3).to(device)
model = DeepGCNWithResidual(input_dim=376, hidden_dim=128, output_dim=3).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# 初始化损失记录列表（训练+验证）
train_losses = []
val_losses = []  # 验证损失记录

# train loop
for epoch in range(5000):
    # -------------------- 训练阶段 --------------------
    model.train()
    epoch_train_loss = 0.0
    num_train_batches = 0

    for batch in train_loader:
        optimizer.zero_grad()
        output = model(batch.to(device))
        loss = F.mse_loss(output, batch.y)
        loss.backward()
        optimizer.step()

        epoch_train_loss += loss.item()
        num_train_batches += 1

    avg_train_loss = epoch_train_loss / num_train_batches
    train_losses.append(avg_train_loss)

    # -------------------- 验证阶段 --------------------
    model.eval()  # 切换评估模式
    epoch_val_loss = 0.0
    num_val_batches = 0

    with torch.no_grad():  # 禁用梯度计算节省内存
        for val_batch in val_loader:
            output = model(val_batch.to(device))
            val_batch.y = val_batch.y
            val_loss = F.mse_loss(output, val_batch.y)
            epoch_val_loss += val_loss.item()
            num_val_batches += 1

    avg_val_loss = epoch_val_loss / num_val_batches
    val_losses.append(avg_val_loss)

    # 打印训练和验证损失（每50轮打印一次）
    if epoch % 50 == 0:
        print(f"Epoch {epoch}:")
        print("output:", output[-1])
        print("labels: ", val_batch.y[-1])
        print(f"  Train Loss: {avg_train_loss:.6f}")
        print(f"  Val Loss:   {avg_val_loss:.6f}\n")

# 绘制双损失曲线
plt.figure(figsize=(10, 6))
plt.plot(range(100, len(train_losses)+1), train_losses[99:], 'b-', label='Training Loss')
plt.plot(range(100, len(val_losses)+1), val_losses[99:], 'r--', label='Validation Loss')
plt.title('Training & Validation Loss Curves', fontsize=14)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('MSE Loss', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.savefig('loss_curves.png')
plt.show()