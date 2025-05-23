import numpy as np
from sklearn.decomposition import PCA

def svd_feature(mask: np.ndarray, k: int = 30) -> np.ndarray:
    """
    对单张 225×225 掩码做 SVD，返回前 k 个奇异值作为特征向量。
    参数：
        mask: 二值矩阵，shape=(225,225)，dtype 0/1
        k:    取前 k 个奇异值
    返回：
        feat: shape=(k,) 的 numpy 向量
    """
    # 1. 计算 SVD
    U, S, Vt = np.linalg.svd(mask.astype(float), full_matrices=False)
    # 2. 取前 k 个奇异值
    # feat = S[:k]
    feat = S
    return feat


if __name__ == "__main__":
    # 测试
    # 随机生成一张二值掩码
    mask = (np.random.rand(255, 255) > 0.7).astype(int)
    feat_svd = svd_feature(mask, k=30)
    print("SVD 特征向量维度：", feat_svd)  # (255,)
