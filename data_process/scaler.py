import torch

class TargetScaler:
    """
    用于对三维目标向量(y)进行均值-方差归一化（standardization）。
    - fit(all_targets): 计算并保存每个维度的 mean 和 std。
    - transform(y):      将输入 y 归一化到 (y - mean) / std。
    - inverse_transform(y_norm): 将归一化后的 y_norm 还原回原始尺度：y_norm * std + mean。
    """

    def __init__(self, eps: float = 1e-6):
        # eps 用于防止除以 0
        self.mean = None    # 会保存形状为 [3] 的均值张量
        self.std = None     # 会保存形状为 [3] 的标准差张量
        self.eps = eps

    def fit(self, all_targets: torch.Tensor):
        """
        传入所有样本的目标值张量：shape [N, 3]。
        计算并保存每个维度的均值(mean)和标准差(std)。
        """
        # all_targets 形状假设为 [N, 3]
        if not isinstance(all_targets, torch.Tensor):
            raise ValueError("输入必须是 torch.Tensor，且形状为 [N, 3]")
        if all_targets.dim() != 2 or all_targets.size(1) != 3:
            raise ValueError("输入 all_targets 形状必须为 [N, 3]")

        # 计算每一列(维度)的均值和标准差
        # dim=0: 对 N 个样本做均值/标准差
        self.mean = all_targets.mean(dim=0)           # 形状 [3]
        self.std  = all_targets.std(dim=0, unbiased=False)  # 形状 [3]
        # 防止某一维 std 为 0
        self.std = torch.where(self.std < self.eps, 
                               torch.ones_like(self.std), 
                               self.std)

    def transform(self, y: torch.Tensor) -> torch.Tensor:
        """
        对输入 y 做归一化，y 的形状应为 [batch_size, 3] 或 [3]。
        返回 (y - mean) / std，形状与 y 相同。
        """
        if self.mean is None or self.std is None:
            raise RuntimeError("必须先对所有样本调用 fit()，再调用 transform()。")

        # 如果 y 是一维向量 [3]，就先 unsqueeze(0) 变为 [1, 3]
        was_1d = False
        if y.dim() == 1 and y.size(0) == 3:
            y = y.unsqueeze(0)
            was_1d = True

        if y.dim() != 2 or y.size(1) != 3:
            raise ValueError("y 的形状必须为 [batch_size, 3] 或 [3]")

        # 为了操作方便，把 mean/std 扩展到 [batch_size, 3]
        mean = self.mean.unsqueeze(0).to(y.device)   # [1, 3]
        std  = self.std.unsqueeze(0).to(y.device)    # [1, 3]

        y_norm = (y - mean) / std  # 广播机制

        if was_1d:
            return y_norm.squeeze(0)
        return y_norm

    def inverse_transform(self, y_norm: torch.Tensor) -> torch.Tensor:
        """
        将归一化后的 y_norm 还原回原始尺度：
        y_orig = y_norm * std + mean。
        支持 y_norm 形状为 [batch_size, 3] 或 [3]，返回形状与之相同。
        """
        if self.mean is None or self.std is None:
            raise RuntimeError("必须先对所有样本调用 fit()，再调用 inverse_transform()。")

        was_1d = False
        if y_norm.dim() == 1 and y_norm.size(0) == 3:
            y_norm = y_norm.unsqueeze(0)
            was_1d = True

        if y_norm.dim() != 2 or y_norm.size(1) != 3:
            raise ValueError("y_norm 的形状必须为 [batch_size, 3] 或 [3]")

        mean = self.mean.unsqueeze(0).to(y_norm.device)
        std  = self.std.unsqueeze(0).to(y_norm.device)

        y_orig = y_norm * std + mean

        if was_1d:
            return y_orig.squeeze(0)
        return y_orig
