# GNN
# 图神经网络库（Graph Neural Network Library）

本项目提供了基于 PyTorch 的图神经网络（GNN）库，实现了常见的 GCN、GAT 和 GraphSAGE 模型，并支持自定义扩展。主要功能包括：

- **模型支持**：内置 GCN、GAT 和 GraphSAGE 等模型结构，可直接使用或在此基础上扩展自定义模型。
- **模块化设计**：网络结构可灵活组合层、激活函数、Dropout 等模块，方便用户自定义模型。
- **数据适配**：支持从 CSV/JSON 文件加载图数据（边列表、节点特征、标签等），自动转换为 PyTorch 数据结构。
- **预处理工具**：提供邻接矩阵和特征归一化、数据集划分等常用预处理功能。
- **使用示例**：附带示例代码，演示如何导入数据、定义模型并进行训练与评估。

## 快速开始

1. 安装依赖：
   - Python 3.x
   - PyTorch

2. 导入模块：
   ```python
   from graphnn.data.reader import load_graph_from_csv
   from graphnn.data.preprocessing import normalize_features, normalize_adjacency, train_val_test_split
   from graphnn.models.gcn import GCN
   from graphnn.train import train_model, test_model
