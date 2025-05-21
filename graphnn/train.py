# -*- coding: utf-8 -*-
"""训练与评估模块：包含模型训练和测试的辅助函数"""
import torch
import torch.nn.functional as F

def train_model(model, graph_data, epochs=100, lr=0.01, weight_decay=5e-4):
    """训练GNN模型，并返回每个Epoch的训练损失和准确率列表"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_list = []
    acc_list = []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(graph_data.features, graph_data.adjacency)
        if graph_data.labels is None or graph_data.train_mask is None:
            raise ValueError('缺少标签或训练集划分信息')
        loss = F.cross_entropy(output[graph_data.train_mask], graph_data.labels[graph_data.train_mask])
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())
        # 计算训练集准确率
        pred = output.argmax(dim=1)
        correct = (pred[graph_data.train_mask] == graph_data.labels[graph_data.train_mask]).sum().item()
        acc = correct / graph_data.train_mask.sum().item()
        acc_list.append(acc)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}, Train Acc: {acc:.4f}')
    return loss_list, acc_list

def test_model(model, graph_data):
    """在测试集上评估模型准确率"""
    model.eval()
    with torch.no_grad():
        output = model(graph_data.features, graph_data.adjacency)
        if graph_data.labels is None or graph_data.test_mask is None:
            raise ValueError('缺少标签或测试集划分信息')
        pred = output.argmax(dim=1)
        correct = (pred[graph_data.test_mask] == graph_data.labels[graph_data.test_mask]).sum().item()
        acc = correct / graph_data.test_mask.sum().item()
        print(f'Test Accuracy: {acc:.4f}')
    return acc
