import torch
import numpy as np
from sklearn.preprocessing import StandardScaler

def normalize(batch_y):
    # 2. 分维度标准化（每个维度独立处理）
    scalers = []  # 存储每个维度的标准化器
    y_normalized = np.zeros_like(batch_y.cpu())

    for i in range(batch_y.shape[1]):  # 遍历每个维度
        scaler = StandardScaler()
        y_normalized[:, i] = scaler.fit_transform(batch_y[:, i].reshape(-1, 1)).flatten()
        scalers.append(scaler)  # 保存标准化器
    y_normalized = torch.tensor(y_normalized)

    return y_normalized, scalers



def restore(y_normalized, scalers):
    # 3. 逆变换恢复原始维度
    y_restored = np.zeros_like(y_normalized)

    for i in range(y_normalized.shape[1]):
        y_restored[:, i] = scalers[i].inverse_transform(y_normalized[:, i].reshape(-1, 1)).flatten()
    
    return y_restored





if __name__ == '__main__':

    # 1. 原始三维标签数据
    y_original = torch.tensor([
        [0.01, 0.2, 9.8],
        [0.03, 0.5, 10.2],
        [0.02, 0.3, 8.5]
    ])

    y_normalized, scalers = normalize(y_original)
    print('Normalized y:', y_normalized)

    y_restored = restore(y_normalized, scalers)
    print('Restored y:', y_restored)
