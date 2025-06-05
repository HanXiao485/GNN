from config import GraphDataset
from gnn_model import GNN, HybridGCN_SAGE, DeepGCNWithResidual, GATWithAttentionPool, GINWithSet2Set
from data_process import TargetScaler

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader as GeoDataLoader
from torch.utils.data import random_split

import matplotlib.pyplot as plt

# -------------------- 1. 先加载整个数据集，并提取所有 target 以拟合 TargetScaler --------------------
JSON_PATH = "datas/result.json"
full_dataset = GraphDataset(JSON_PATH)

# 假设 GraphDataset.__getitem__ 返回的 data.y 是一个形状为 [3] 的 torch.Tensor（三维目标向量）。
# 我们先把所有样本的 y 收集出来，堆叠成一个 [N, 3] 的大 tensor。
all_targets = []
for data in full_dataset:
    # data.y 形状假设为 torch.Tensor([y1, y2, y3])
    all_targets.append(data.y)  # 变为 [1, 3]
all_targets = torch.cat(all_targets, dim=0)  # 变为 [N, 3]

# 创建并拟合一个 TargetScaler，只用来对“目标值”做归一化
target_scaler = TargetScaler()
# 如果 TargetScaler 要的是 numpy 数组，也可以改成 all_targets.numpy()
target_scaler.fit(all_targets)  # 拟合均值和方差等

# -------------------- 2. 划分训练/验证集（8:2） --------------------
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# -------------------- 3. 数据加载器 --------------------
train_loader = GeoDataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader   = GeoDataLoader(val_dataset,   batch_size=128, shuffle=False)

# -------------------- 4. 模型 + 优化器 --------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = DeepGCNWithResidual(input_dim=376, hidden_dim=32, output_dim=3).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# -------------------- 5. 训练／验证 循环（带 target 归一化） --------------------
train_losses = []
val_losses   = []

for epoch in range(5000):
    # ---------- 5.1 训练阶段 ----------
    model.train()
    epoch_train_loss = 0.0
    num_train_batches = 0

    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        # ---- 对 batch.y 做“归一化” ----
        # 注意：TargetScaler 可能要求输入是 CPU 上的 numpy 或 tensor，如果需要，请先 cpu() 再转 numpy()。
        # 假设 target_scaler.transform 可以接受一个 torch.Tensor 并返回归一化后的 torch.Tensor。
        y_true = batch.y  # 形状 [batch_size, 3]
        y_norm = target_scaler.transform(y_true)  # 归一化后的 target

        # 把归一化后的目标值放回 batch.y
        batch.y = y_norm

        # 前向+反向
        output_norm = model(batch)  # output_norm 也是归一化空间里的预测值，[batch_size, 3]
        loss = F.mse_loss(output_norm, batch.y)
        loss.backward()
        optimizer.step()

        epoch_train_loss += loss.item()
        num_train_batches += 1

    avg_train_loss = epoch_train_loss / num_train_batches
    train_losses.append(avg_train_loss)

    # ---------- 5.2 验证阶段 ----------
    model.eval()
    epoch_val_loss = 0.0
    num_val_batches = 0

    with torch.no_grad():
        for val_batch in val_loader:
            val_batch = val_batch.to(device)

            # 同样要把 val_batch.y 归一化
            y_val_true = val_batch.y
            y_val_norm = target_scaler.transform(y_val_true)
            val_batch.y = y_val_norm

            output_val_norm = model(val_batch)
            val_loss = F.mse_loss(output_val_norm, val_batch.y)
            epoch_val_loss += val_loss.item()
            num_val_batches += 1

    avg_val_loss = epoch_val_loss / num_val_batches
    val_losses.append(avg_val_loss)

    # ---------- 5.3 每 50 个 epoch 打印一次 train/val 信息（同时演示如何“还原回原始尺度”看预测值） ----------
    if epoch % 50 == 0:
        # 取最后一个 validation batch 的结果做演示：
        # output_val_norm 是归一化空间的预测，我们要 inverse_transform 回原始尺度。
        output_last_norm = output_val_norm[-1].detach().cpu()
        output_last_orig = target_scaler.inverse_transform(output_last_norm.unsqueeze(0))[0]

        true_last_orig = y_val_true[-1].detach().cpu()  # 这是原始尺度的标签

        print(f"Epoch {epoch}:")
        print("  Predict (原始尺度):", output_last_orig)
        print("  Label   (原始尺度):", true_last_orig)
        print(f"  Train Loss(norm space): {avg_train_loss:.6f}")
        print(f"  Val   Loss(norm space): {avg_val_loss:.6f}\n")

# -------------------- 6. 画出训练&验证的 loss 曲线 --------------------
plt.figure(figsize=(10, 6))
plt.plot(range(100, len(train_losses) + 1), train_losses[99:], 'b-', label='Training Loss (norm space)')
plt.plot(range(100, len(val_losses)   + 1), val_losses[99:],   'r--', label='Validation Loss (norm space)')
plt.title('Training & Validation Loss Curves (Normalized Targets)', fontsize=14)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('MSE Loss (normalized)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.savefig('loss_curves.png')
plt.show()
