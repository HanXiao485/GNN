import networkx as nx
import torch

def generate_complete_edge_index(n_nodes):
    # 生成无向完全图
    G = nx.complete_graph(n_nodes)
    # 提取边并添加双向连接
    edge_list = []
    for u, v in G.edges():
        edge_list.extend([[u, v], [v, u]])  # 添加双向边
    # 转换为PyTorch张量
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    return edge_index




if __name__ == "__main__":
    # 示例：生成3个节点的边索引
    edge_index = generate_complete_edge_index(3)
    print(edge_index)