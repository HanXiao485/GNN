import numpy as np

def update_pic_info(labels, masks, pic_info):
    """
    根据 labels 和 masks 更新 pic_info 字典。

    参数:
        labels (list of str): 标签列表，例如 ['1', '1', '2']
        masks (np.ndarray): 布尔类型或 0/1 数组，形状为 (N, W)
        pic_info (dict): 初始化的 pic_info 字典，格式为 {i: [[0]*W, [0]*W] for i in range(201)}

    返回:
        dict: 更新后的 pic_info 字典
    """
    # 转换 labels 为整数列表
    labels_int = [int(l) for l in labels]

    # 转换 masks 为 uint8 类型（True->1, False->0）
    masks_uint8 = masks.astype(np.uint8)

    # 获取宽度（列数）
    width = masks_uint8.shape[1]

    # 遍历标签和掩码并更新 pic_info
    for label, mask in zip(labels_int, masks_uint8):
        # 增加计数
        count = pic_info[label][0][0] + 1
        pic_info[label][0] = [count] * width

        # 累加掩码（逐元素加）
        pic_info[label][1] = [pic_info[label][1][j] + int(mask[j]) for j in range(width)]

    return pic_info


if __name__ == '__main__':
    # 示例数据
    labels = ['1', '1', '2']
    masks = np.array([[False, True,  True,  False, True, True, True],
                    [False, True,  True,  False, True, True, True],
                    [False, True,  True,  False, True, True, True]])

    # 初始化 pic_info 字典
    pic_info = {i: [[0]*masks.shape[1], [0]*masks.shape[1]] for i in range(201)}

    # 调用函数
    pic_info = update_pic_info(labels, masks, pic_info)

    # 打印结果
    print("Label 1:", pic_info[1])
    print("Label 2:", pic_info[2])
